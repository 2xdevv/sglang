"""Host-pool allocator whose buffers 3FS USRBIO can read into and write from.

USRBIO only transfers to/from memory the 3FS FUSE daemon has mapped itself:
`make_iovec()` symlinks `/dev/shm/<name>` into the mount and the daemon
`shm_open`s that name, maps it, and registers it for RDMA. The host pool must
therefore be backed by a `/dev/shm` file from the start; wrapping an anonymous
mapping after the fact (or after `cudaHostRegister` pinned it) is not enough.

Each allocation is a named `/dev/shm` file whose name exists only until its
first successful registration (or process exit), which bounds what a crash can
leak to the window between pool allocation and storage attach. The FUSE
mapping keeps the pages alive without the name, but a later registration
(storage detached and attached again) cannot reopen it and uses staging
copies instead. Allocation ownership stays here; registrations are owned by
the storage backend and must be dropped before the host pool is freed.
"""

from __future__ import annotations

import logging
import math
import mmap
import os
import threading
import uuid
import weakref
from contextlib import contextmanager
from typing import Iterator, List

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host.common import HostTensorAllocator
from sglang.srt.mem_cache.storage.mmap.mmap_allocator import _mmap_prefaulted

logger = logging.getLogger(__name__)

SHM_DIR = "/dev/shm"


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


class Hf3fsShmAllocation:
    """One `/dev/shm`-backed mapping that may be registered with 3FS USRBIO once."""

    def __init__(self, nbytes: int, shm_dir: str = SHM_DIR):
        if nbytes <= 0:
            raise ValueError(f"allocation size must be positive, got {nbytes}")
        # The FUSE daemon maps the whole file (st_size), so the file length and
        # the local mapping length must agree.
        self.size = math.ceil(nbytes / mmap.PAGESIZE) * mmap.PAGESIZE

        stat = os.statvfs(shm_dir)
        available = stat.f_bavail * stat.f_frsize
        if self.size > available:
            raise OSError(
                f"{shm_dir} has {available / 1e9:.2f} GB free but the host pool "
                f"buffer needs {self.size / 1e9:.2f} GB"
            )

        self.name = f"sglang-hf3fs-{os.getpid()}-{uuid.uuid4().hex}"
        path = os.path.join(shm_dir, self.name)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        # Also runs at interpreter exit for a pool that was never registered.
        self._name_finalizer = weakref.finalize(self, _unlink_quietly, path)
        try:
            os.ftruncate(fd, self.size)
            self.mm = _mmap_prefaulted(fd, self.size, mmap.MAP_SHARED)
        except BaseException:
            self._name_finalizer()
            raise
        finally:
            os.close(fd)
        self._name_lock = threading.Lock()
        self.base_ptr = torch.frombuffer(self.mm, dtype=torch.uint8, count=1).data_ptr()

    def tensor(self, dims: tuple, dtype: torch.dtype) -> torch.Tensor:
        count = math.prod(dims)
        return torch.frombuffer(self.mm, dtype=dtype, count=count).reshape(dims)

    @contextmanager
    def registration_name(self) -> Iterator[str]:
        """Yield the `/dev/shm` name to register; remove it once that succeeds.

        By then the FUSE daemon has mapped the file. A failed registration keeps
        the name so a later attach can retry.
        """
        with self._name_lock:
            if not self._name_finalizer.alive:
                raise RuntimeError(
                    "this host buffer was already registered with 3FS once and its "
                    "/dev/shm name is gone; it cannot be registered again"
                )
            yield self.name
            self._name_finalizer()


class Hf3fsHostTensorAllocator(HostTensorAllocator):
    """Allocate host-pool tensors in `/dev/shm` so HF3FS can do zero-copy I/O.

    Falls back to the default anonymous mmap (and therefore to the staging-copy
    I/O path) for any buffer that cannot be placed in `/dev/shm`.
    """

    def __init__(self, shm_dir: str = SHM_DIR):
        super().__init__()
        self.shm_dir = shm_dir
        self.allocations: List[Hf3fsShmAllocation] = []

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert device == "cpu", (
            f"Hf3fsHostTensorAllocator only supports CPU allocations; got device={device!r}"
        )
        self.dtype = dtype
        self.dims = dims
        if (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip():
            logger.warning(
                "SGLANG_HUGEPAGE_SIZE is ignored by the HF3FS zero-copy host "
                "allocator: %s is not a hugetlbfs mount.",
                self.shm_dir,
            )
        nbytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()
        try:
            allocation = Hf3fsShmAllocation(nbytes, self.shm_dir)
        except OSError as e:
            logger.warning(
                "HF3FS zero-copy host allocation of %.2f GB failed (%s); this "
                "buffer uses anonymous memory and HF3FS I/O on it falls back to "
                "staging copies.",
                nbytes / 1e9,
                e,
            )
            return super().allocate(dims, dtype, device)
        self.allocations.append(allocation)
        logger.info(
            "Allocated %.2f GB HiCache host buffer in %s for HF3FS zero-copy I/O.",
            allocation.size / 1e9,
            self.shm_dir,
        )
        return allocation.tensor(dims, dtype)
