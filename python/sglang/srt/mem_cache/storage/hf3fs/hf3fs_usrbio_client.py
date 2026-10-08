import bisect
import datetime
import errno
import functools
import logging
import multiprocessing.shared_memory
import os
import threading
import time
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

import msgspec
import torch

from sglang.srt.mem_cache.storage.hf3fs.hf3fs_client import Hf3fsClient
from sglang.srt.mem_cache.storage.hf3fs.hf3fs_host_allocator import (
    Hf3fsShmAllocation,
)
from sglang.srt.utils.cpp_extension_loader import load_extension_with_recovery

root = Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)

HF3FS_AVAILABLE = True
try:
    from hf3fs_fuse.io import (
        deregister_fd,
        extract_mount_point,
        make_ioring,
        make_iovec,
        register_fd,
    )
except ImportError:
    HF3FS_AVAILABLE = False

# 3FS registers each iov with RDMA in blocks of this size and rejects an I/O
# that crosses a block; one block per pool can exceed the IB driver's MR limit.
# Arbitrary; 1 GiB matches the buffer size 3FS's own read_file() uses.
DEFAULT_IOV_BLOCK_BYTES = 1 << 30


@functools.cache
def _hf3fs_utils():
    return load_extension_with_recovery(
        name="hf3fs_utils", sources=[f"{root}/hf3fs_utils.cpp"]
    )


def rsynchronized():
    def _decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.rlock:
                return func(self, *args, **kwargs)

        return wrapper

    return _decorator


def wsynchronized():
    def _decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.wlock:
                return func(self, *args, **kwargs)

        return wrapper

    return _decorator


def iov_block_bytes(io_bytes: int, region_bytes: int, target_bytes: int) -> int:
    """Largest multiple of `io_bytes` not above `target_bytes`; 0 means one block."""
    if io_bytes <= 0:
        raise ValueError(f"io_bytes must be positive, got {io_bytes}")
    block = max(target_bytes // io_bytes, 1) * io_bytes
    # 3FS rejects a block size larger than the registered file.
    return 0 if block >= region_bytes else block


class _DirectIovRegion(msgspec.Struct, frozen=True):
    base_ptr: int
    size: int
    block_size: int
    iov: object
    allocation: Hf3fsShmAllocation


class Hf3fsDirectIovs:
    """USRBIO registrations of host-pool allocations, shared by every client ring.

    Owned by the storage backend, not by the host pool: `close()` drops the 3FS
    registrations but leaves the allocations (and the live HiCache pool) intact.
    """

    def __init__(
        self,
        mount_point: str,
        allocations: Sequence[Hf3fsShmAllocation],
        io_bytes: int,
        target_block_bytes: int = DEFAULT_IOV_BLOCK_BYTES,
    ):
        if not HF3FS_AVAILABLE:
            raise ImportError("hf3fs_fuse.io is not available.")
        regions = []
        for allocation in allocations:
            block_size = iov_block_bytes(
                io_bytes=io_bytes,
                region_bytes=allocation.size,
                target_bytes=target_block_bytes,
            )
            with allocation.registration_name() as name:
                shm = SimpleNamespace(name=name, buf=memoryview(allocation.mm))
                # numa=-1: binding would migrate pages cudaHostRegister pinned.
                iov = make_iovec(shm, mount_point, block_size=block_size, numa=-1)
            regions.append(
                _DirectIovRegion(
                    base_ptr=allocation.base_ptr,
                    size=allocation.size,
                    block_size=block_size,
                    iov=iov,
                    allocation=allocation,
                )
            )
        regions.sort(key=lambda r: r.base_ptr)
        self._regions = regions
        self._bases = [r.base_ptr for r in regions]
        self._warned_block_cross = False
        logger.info(
            "Registered %d HiCache host buffer(s) (%.2f GB) with 3FS USRBIO at %s.",
            len(regions),
            sum(r.size for r in regions) / 1e9,
            mount_point,
        )

    def slice_for(self, tensor: torch.Tensor):
        """The registered iov slice covering `tensor`, or None to use staging."""
        if self._regions is None or not tensor.is_contiguous():
            return None
        ptr = tensor.data_ptr()
        nbytes = tensor.numel() * tensor.element_size()
        i = bisect.bisect_right(self._bases, ptr) - 1
        if i < 0 or nbytes == 0:
            return None
        region = self._regions[i]
        offset = ptr - region.base_ptr
        if offset + nbytes > region.size:
            return None
        bs = region.block_size
        if bs and offset // bs != (offset + nbytes - 1) // bs:
            if not self._warned_block_cross:
                self._warned_block_cross = True
                logger.warning(
                    "HF3FS I/O of %d bytes at buffer offset %d crosses a %d-byte "
                    "registration block; such I/Os use staging copies.",
                    nbytes,
                    offset,
                    bs,
                )
            return None
        return region.iov[offset : offset + nbytes]

    def close(self) -> None:
        # Dropping the iovec unlinks its 3FS symlink, which deregisters the buffer.
        self._regions = None
        self._bases = []


class Hf3fsUsrBioClient(Hf3fsClient):
    """HF3FS client implementation using usrbio."""

    def __init__(
        self,
        path: str,
        size: int,
        bytes_per_page: int,
        entries: int,
        client_timeout: int,
    ):
        if not HF3FS_AVAILABLE:
            raise ImportError(
                "hf3fs_fuse.io is not available. Please install the hf3fs_fuse package."
            )

        self.path = path
        self.size = size
        self.bytes_per_page = bytes_per_page
        self.entries = entries
        self.client_timeout = client_timeout
        self.direct_iovs: Optional[Hf3fsDirectIovs] = None
        self._closed = False
        # Built here, not on first I/O: worker threads would race the JIT build.
        self.hf3fs_utils = _hf3fs_utils()

        self.file = os.open(self.path, os.O_RDWR | os.O_CREAT)
        os.ftruncate(self.file, size)
        register_fd(self.file)

        self.hf3fs_mount_point = extract_mount_point(path)
        self.bs = self.bytes_per_page
        # Staging for tensors outside registered host memory.
        self.shm_r = multiprocessing.shared_memory.SharedMemory(
            size=self.bs * self.entries, create=True
        )
        self.shm_w = multiprocessing.shared_memory.SharedMemory(
            size=self.bs * self.entries, create=True
        )

        self.shm_r_tensor = torch.frombuffer(self.shm_r.buf, dtype=torch.uint8)
        self.shm_w_tensor = torch.frombuffer(self.shm_w.buf, dtype=torch.uint8)

        self.numa = -1
        self.ior_r = make_ioring(
            self.hf3fs_mount_point,
            self.entries,
            for_read=True,
            timeout=1,
            numa=self.numa,
        )
        self.ior_w = make_ioring(
            self.hf3fs_mount_point,
            self.entries,
            for_read=False,
            timeout=1,
            numa=self.numa,
        )
        self.iov_r = make_iovec(self.shm_r, self.hf3fs_mount_point)
        self.iov_w = make_iovec(self.shm_w, self.hf3fs_mount_point)
        self.shm_r.unlink()
        self.shm_w.unlink()

        self.rlock = threading.RLock()
        self.wlock = threading.RLock()

    def set_direct_iovs(self, direct_iovs: Optional[Hf3fsDirectIovs]) -> None:
        with self.rlock, self.wlock:
            self.direct_iovs = direct_iovs

    @rsynchronized()
    def batch_read(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:
        self.check(offsets, tensors)
        iovs, staged = self._assign_iovs(tensors, self.iov_r)
        results = self._run_batch(self.ior_r, True, offsets, iovs)
        if staged:
            try:
                self.hf3fs_utils.read_shm(
                    self.shm_r_tensor, [tensors[i] for i in staged]
                )
            except Exception as e:
                logger.error(f"[Hf3fsUsrBioClient] read_shm failed: {e}", exc_info=True)
                for i in staged:
                    results[i] = 0
        return results

    @wsynchronized()
    def batch_write(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:
        self.check(offsets, tensors)
        iovs, staged = self._assign_iovs(tensors, self.iov_w)
        if staged:
            self.hf3fs_utils.write_shm([tensors[i] for i in staged], self.shm_w_tensor)
        return self._run_batch(self.ior_w, False, offsets, iovs)

    def _assign_iovs(self, tensors: List[torch.Tensor], staging_iov):
        """Point each I/O at its tensor in place, or at the next staging slot."""
        iovs, staged = [], []
        staging_offset = 0
        for i, tensor in enumerate(tensors):
            iov = None
            if self.direct_iovs is not None:
                iov = self.direct_iovs.slice_for(tensor)
            if iov is None:
                size = tensor.numel() * tensor.itemsize
                iov = staging_iov[staging_offset : staging_offset + size]
                staging_offset += size
                staged.append(i)
            iovs.append(iov)
        return iovs, staged

    def _run_batch(self, ior, read: bool, offsets: List[int], iovs) -> List[int]:
        """Prepare, submit and fully drain one batch; 0 marks a failed I/O.

        Never returns while an I/O it prepared may still touch its buffer:
        prepare() can start an I/O before submit(), and wait() may return early.
        """
        results = [0] * len(offsets)
        prepared = 0
        for i, (offset, iov) in enumerate(zip(offsets, iovs)):
            try:
                ior.prepare(iov, read, self.file, offset, userdata=i)
            except Exception as e:
                logger.error(
                    f"Error preparing batch {'read' if read else 'write'}: {e}"
                )
                break
            prepared += 1
        if prepared == 0:
            return results
        try:
            ior.submit()
        except Exception as e:
            # The I/O worker still discovers prepared entries without a submit.
            logger.error(f"Error submitting batch {'read' if read else 'write'}: {e}")
        self._drain(ior, prepared, results)
        return results

    def _drain(self, ior, prepared: int, results: List[int]) -> None:
        pending = set(range(prepared))
        timeout = datetime.timedelta(seconds=self.client_timeout)
        deadline = time.monotonic() + self.client_timeout
        timed_out = False
        logged_wait_error = False
        while pending:
            try:
                completions = ior.wait(min_results=len(pending), timeout=timeout)
            except Exception as e:
                # wait() raises ETIMEDOUT when nothing completed in time.
                completions = []
                if not (isinstance(e, OSError) and e.errno == errno.ETIMEDOUT):
                    if not logged_wait_error:
                        logged_wait_error = True
                        logger.error(f"Error waiting for HF3FS I/O: {e}")
                    time.sleep(0.001)
            for completion in completions:
                index = completion.userdata
                if index not in pending:
                    logger.error(f"Unexpected HF3FS completion userdata={index!r}")
                    continue
                pending.discard(index)
                # A late completion keeps the client_timeout failure semantics.
                results[index] = 0 if timed_out else completion.result
            if pending and not timed_out and time.monotonic() >= deadline:
                timed_out = True
                logger.error(
                    f"{len(pending)} HF3FS I/O(s) still outstanding after "
                    f"{self.client_timeout}s; waiting for them before releasing "
                    "their buffers."
                )

    def check(self, offsets: List[int], tensors: List[torch.Tensor]) -> None:
        sizes = [t.numel() * t.itemsize for t in tensors]
        if any(
            [
                len(offsets) > self.entries,
                len(offsets) != len(sizes),
                any(
                    [
                        offset < 0 or offset + size > self.size
                        for offset, size in zip(offsets, sizes)
                    ]
                ),
                any([size > self.bytes_per_page for size in sizes]),
            ]
        ):
            raise ValueError(f"Hf3fsClient.check: {offsets=}, {sizes=}")

    def get_size(self) -> int:
        return self.size

    def close(self) -> None:
        # The locks serialize with in-flight batches, which drain before returning.
        with self.rlock, self.wlock:
            if self._closed:
                return
            self._closed = True
            deregister_fd(self.file)
            os.close(self.file)
            self.direct_iovs = None
            del self.ior_r
            del self.ior_w
            del self.iov_r
            del self.iov_w
            # SharedMemory.close() raises BufferError while a tensor exports it.
            del self.shm_r_tensor
            del self.shm_w_tensor
            self.shm_r.close()
            self.shm_w.close()

    def flush(self) -> None:
        os.fsync(self.file)
