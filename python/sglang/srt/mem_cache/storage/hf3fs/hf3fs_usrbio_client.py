import datetime
import logging
import multiprocessing
import os
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Callable, List, Optional

import torch

from sglang.srt.mem_cache.storage.hf3fs.hf3fs_client import Hf3fsClient
from sglang.srt.utils.cpp_extension_loader import load_extension_with_recovery

root = Path(__file__).parent.resolve()
hf3fs_utils = load_extension_with_recovery(
    name="hf3fs_utils", sources=[f"{root}/hf3fs_utils.cpp"]
)

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

        self.file = os.open(self.path, os.O_RDWR | os.O_CREAT)
        os.ftruncate(self.file, size)
        register_fd(self.file)

        self.hf3fs_mount_point = extract_mount_point(path)
        self.bs = self.bytes_per_page
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

    @rsynchronized()
    def batch_read(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:
        self.check(offsets, tensors)
        results = [0] * len(offsets)
        # Each page has its own shm region until all prepared reads complete.
        # Reap ready pages without waiting for the whole batch so their copy
        # into the destination tensors overlaps the remaining reads.
        num_prepared = 0
        shm_offsets = []
        current = 0
        try:
            for offset, tensor in zip(offsets, tensors):
                size = tensor.numel() * tensor.itemsize
                shm_offsets.append(current)
                self.ior_r.prepare(
                    self.iov_r[current : current + size],
                    True,
                    self.file,
                    offset,
                    userdata=num_prepared,
                )
                num_prepared += 1
                current += size
        except Exception as e:
            logger.error(f"Error preparing batch read: {e}")

        if num_prepared:
            try:
                self.ior_r.submit()
            except Exception as e:
                logger.error(f"Error submitting batch read: {e}")

        def copy_page(index: int, result: int) -> None:
            tensor = tensors[index]
            size = tensor.numel() * tensor.itemsize
            # Failed and short reads must not copy stale bytes from shm.
            if result == size:
                start = shm_offsets[index]
                hf3fs_utils.read_shm(self.shm_r_tensor[start : start + size], [tensor])

        self._wait_all(self.ior_r, num_prepared, results, "read", copy_page)
        return results

    @wsynchronized()
    def batch_write(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:
        self.check(offsets, tensors)
        results = [0] * len(offsets)
        # prepare
        hf3fs_utils.write_shm(tensors, self.shm_w_tensor)
        current = 0
        for offset, tensor in zip(offsets, tensors):
            size = tensor.numel() * tensor.itemsize
            try:
                self.ior_w.prepare(
                    self.iov_w[current : current + size], False, self.file, offset
                )
                current += size
            except Exception as e:
                logger.error(f"Error preparing batch write: {e}")
                return results

        # submit
        ionum = len(offsets)
        try:
            resv = self.ior_w.submit().wait(
                min_results=ionum,
                timeout=datetime.timedelta(seconds=self.client_timeout),
            )
        except Exception as e:
            logger.error(f"Error submitting batch write: {e}")
            return results

        # results
        results = [res.result for res in resv]

        return results

    def _wait_all(
        self,
        ior,
        num_ios: int,
        results: List[int],
        op: str,
        on_complete: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        # Returning with I/O in flight would let the next call overwrite its shm
        # region and ring slot, so wait out every prepared I/O, even past the timeout.
        # Completions arrive in completion order; userdata is the page index.
        timeout = datetime.timedelta(seconds=self.client_timeout)
        done = 0
        while done < num_ios:
            try:
                resv = ior.wait(
                    max_results=num_ios - done,
                    min_results=1 if on_complete is not None else num_ios - done,
                    timeout=timeout,
                )
            except Exception as e:
                logger.error(f"Error waiting for batch {op}: {e}")
                resv = []
                # Avoid a hot loop if wait() fails immediately.
                time.sleep(0.1)
            for res in resv:
                results[res.userdata] = res.result
                if on_complete is not None:
                    try:
                        on_complete(res.userdata, res.result)
                    except Exception as e:
                        results[res.userdata] = 0
                        logger.error(
                            f"[Hf3fsUsrBioClient] batch {op} completion failed: {e}",
                            exc_info=True,
                        )
            done += len(resv)
            if done < num_ios and (not resv or on_complete is None):
                logger.warning(
                    f"[Hf3fsUsrBioClient] batch {op}: {num_ios - done} I/O(s) "
                    f"still outstanding after {self.client_timeout}s, waiting"
                )

    def check(self, offsets: List[int], tensors: List[torch.Tensor]) -> None:
        sizes = [t.numel() * t.itemsize for t in tensors]
        if any(
            [
                len(offsets) > self.entries,
                len(offsets) != len(sizes),
                all(
                    [
                        offset < 0 or offset + size > self.size
                        for offset, size in zip(offsets, sizes)
                    ]
                ),
                all([size > self.bytes_per_page for size in sizes]),
            ]
        ):
            self.close()
            raise ValueError(f"Hf3fsClient.check: {offsets=}, {sizes=}")

    def get_size(self) -> int:
        return self.size

    def close(self) -> None:
        deregister_fd(self.file)
        os.close(self.file)
        del self.ior_r
        del self.ior_w
        del self.iov_r
        del self.iov_w
        self.shm_r.close()
        self.shm_w.close()

    def flush(self) -> None:
        os.fsync(self.file)
