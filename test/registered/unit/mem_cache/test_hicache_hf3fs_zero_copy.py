"""Unit tests for HF3FS zero-copy host-pool I/O (no 3FS mount, no GPU).

A fake USRBIO stands in for the 3FS FUSE daemon: like the daemon, it maps the
iov's /dev/shm name itself, so a test only passes when the registered file
really aliases the host-pool tensor and I/O offsets are computed against it.
"""

import errno
import mmap
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.storage.hf3fs import hf3fs_usrbio_client as usrbio
from sglang.srt.mem_cache.storage.hf3fs.hf3fs_host_allocator import (
    Hf3fsHostTensorAllocator,
)
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import HiCacheHF3FS
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 4096


class _FakeIov:
    def __init__(self, buf: memoryview, block_size: int, root=None, offset: int = 0):
        self.buf = buf
        self.block_size = block_size
        self.root = root if root is not None else self
        self.offset = offset

    def __getitem__(self, s: slice) -> "_FakeIov":
        start, stop, _ = s.indices(len(self.buf))
        return _FakeIov(self.buf[start:stop], self.block_size, self.root, start)


class _FakeRing:
    def __init__(self, usrbio_fake: "_FakeUsrbio", entries: int, for_read: bool):
        self.fake = usrbio_fake
        self.entries = entries
        self.for_read = for_read
        self.queued = []
        self.completed = []

    def prepare(self, iov, read, fd, off, userdata=None):
        assert read == self.for_read
        if self.fake.fail_prepare_after is not None and (
            self.fake.prepare_calls >= self.fake.fail_prepare_after
        ):
            raise OSError(errno.EAGAIN, "injected prepare failure")
        self.fake.prepare_calls += 1
        self.fake.used_iov_roots.append(iov.root)
        io = (iov, fd, off, userdata)
        if self.fake.execute_on_prepare:
            # The I/O worker may pick up prepared entries before submit().
            self.completed.append(self._execute(io, self.for_read))
        else:
            self.queued.append(io)
        return self

    def submit(self):
        if self.fake.fail_submit:
            raise OSError(errno.EIO, "injected submit failure")
        return self

    def wait(self, *, max_results=0, min_results=0, timeout=None):
        self.completed.extend(self._execute(io, self.for_read) for io in self.queued)
        self.queued.clear()
        budget = self.fake.wait_budgets.pop(0) if self.fake.wait_budgets else None
        if budget == 0:
            raise OSError(errno.ETIMEDOUT, "injected timeout")
        take = len(self.completed) if budget is None else budget
        out, self.completed = self.completed[:take], self.completed[take:]
        if self.fake.reverse_completions:
            out.reverse()
        return out

    def outstanding(self) -> int:
        return len(self.queued) + len(self.completed)

    @staticmethod
    def _execute(io, for_read: bool):
        iov, fd, off, userdata = io
        n = len(iov.buf)
        bs = iov.block_size
        if bs and iov.offset // bs != (iov.offset + n - 1) // bs:
            result = -errno.EINVAL
        elif for_read:
            result = os.preadv(fd, [iov.buf], off)
        else:
            result = os.pwrite(fd, iov.buf, off)
        return SimpleNamespace(result=result, userdata=userdata)


class _FakeUsrbio:
    def __init__(self, mount_point: str):
        self.mount_point = mount_point
        self.execute_on_prepare = False
        self.fail_prepare_after = None
        self.fail_submit = False
        self.wait_budgets = []
        self.reverse_completions = False
        self.prepare_calls = 0
        self.used_iov_roots = []
        self.registered_names = []
        self.rings = []

    def make_iovec(self, shm, hf3fs_mount_point, block_size=0, numa=-1):
        # Like the FUSE daemon: open the name and map the whole file ourselves.
        path = f"/dev/shm/{shm.name}"
        fd = os.open(path, os.O_RDWR)
        try:
            size = os.fstat(fd).st_size
            assert len(shm.buf) == size, "iov length must equal the file size"
            assert block_size <= size
            mm = mmap.mmap(fd, size, mmap.MAP_SHARED)
        finally:
            os.close(fd)
        if shm.name.startswith("sglang-hf3fs-"):
            self.registered_names.append(shm.name)
        return _FakeIov(memoryview(mm), block_size)

    def make_ioring(self, hf3fs_mount_point, entries, for_read=True, **kwargs):
        ring = _FakeRing(self, entries, for_read)
        self.rings.append(ring)
        return ring

    def patches(self):
        return mock.patch.multiple(
            usrbio,
            create=True,
            HF3FS_AVAILABLE=True,
            make_iovec=self.make_iovec,
            make_ioring=self.make_ioring,
            register_fd=lambda fd: None,
            deregister_fd=lambda fd: None,
            extract_mount_point=lambda path: self.mount_point,
            _hf3fs_utils=lambda: _PY_HF3FS_UTILS,
        )

    def outstanding(self) -> int:
        return sum(ring.outstanding() for ring in self.rings)


def _bytes_of(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).view(torch.uint8)


def _read_shm(shm: torch.Tensor, dst):
    cur = 0
    for t in dst:
        n = t.numel() * t.element_size()
        _bytes_of(t).copy_(shm[cur : cur + n])
        cur += n


def _write_shm(src, shm: torch.Tensor):
    cur = 0
    for t in src:
        n = t.numel() * t.element_size()
        shm[cur : cur + n].copy_(_bytes_of(t))
        cur += n


_PY_HF3FS_UTILS = SimpleNamespace(read_shm=_read_shm, write_shm=_write_shm)


class _Hf3fsTestBase(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fake = _FakeUsrbio(self.tmp.name)
        patcher = self.fake.patches()
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_client(self, *, entries=8, client_timeout=1, file_pages=64):
        client = usrbio.Hf3fsUsrBioClient(
            path=os.path.join(self.tmp.name, "hicache.bin"),
            size=file_pages * PAGE,
            bytes_per_page=PAGE,
            entries=entries,
            client_timeout=client_timeout,
        )
        self.addCleanup(client.close)
        return client

    def make_direct_iovs(self, allocator, io_bytes=PAGE, target_block_bytes=None):
        return usrbio.Hf3fsDirectIovs(
            mount_point=self.tmp.name,
            allocations=allocator.allocations,
            io_bytes=io_bytes,
            target_block_bytes=target_block_bytes or usrbio.DEFAULT_IOV_BLOCK_BYTES,
        )

    def fill_file_pages(self, client, pages):
        for page_index, value in pages.items():
            os.pwrite(client.file, bytes([value]) * PAGE, page_index * PAGE)


class TestHf3fsDirectIo(_Hf3fsTestBase):
    def test_mha_page_first_direct_reads_k_and_v_in_place(self):
        """K and V pages are separate contiguous I/Os landing in their own halves."""
        allocator = Hf3fsHostTensorAllocator()
        page_num, layer_num, page_size, head_num, head_dim = 4, 2, 4, 2, 128
        kv = allocator.allocate(
            (2, page_num, layer_num, page_size, head_num, head_dim),
            dtype=torch.bfloat16,
            device="cpu",
        )
        self.assertEqual(kv[0, 0].numel() * kv.element_size(), PAGE)
        client = self.make_client()
        direct = self.make_direct_iovs(allocator)
        client.set_direct_iovs(direct)
        self.fill_file_pages(client, {5: 0x11, 9: 0x22})

        page = 2
        results = client.batch_read([5 * PAGE, 9 * PAGE], [kv[0, page], kv[1, page]])

        self.assertEqual(results, [PAGE, PAGE])
        self.assertTrue(torch.all(_bytes_of(kv[0, page]) == 0x11))
        self.assertTrue(torch.all(_bytes_of(kv[1, page]) == 0x22))
        self.assertEqual(int(_bytes_of(kv[:, page - 1]).sum()), 0)
        self.assertEqual(int(_bytes_of(kv[:, page + 1]).sum()), 0)
        self.assertTrue(
            all(root is direct._regions[0].iov for root in self.fake.used_iov_roots)
        )
        direct.close()

    def test_direct_write_sources_reach_file_offsets(self):
        allocator = Hf3fsHostTensorAllocator()
        pages = allocator.allocate((8, PAGE), dtype=torch.uint8, device="cpu")
        pages[3].fill_(0x33)
        pages[6].fill_(0x66)
        client = self.make_client()
        direct = self.make_direct_iovs(allocator)
        client.set_direct_iovs(direct)

        results = client.batch_write([1 * PAGE, 7 * PAGE], [pages[3], pages[6]])

        self.assertEqual(results, [PAGE, PAGE])
        self.assertEqual(os.pread(client.file, PAGE, 1 * PAGE), b"\x33" * PAGE)
        self.assertEqual(os.pread(client.file, PAGE, 7 * PAGE), b"\x66" * PAGE)
        self.assertNotIn(client.iov_w, self.fake.used_iov_roots)
        direct.close()

    def test_unregistered_or_block_crossing_tensors_use_staging(self):
        """An I/O 3FS would reject at a registration block boundary is staged instead."""
        allocator = Hf3fsHostTensorAllocator()
        buf = allocator.allocate((4 * PAGE,), dtype=torch.uint8, device="cpu")
        crossing = buf[PAGE // 2 : PAGE // 2 + PAGE]
        outside = torch.zeros(PAGE, dtype=torch.uint8)
        client = self.make_client()
        direct = self.make_direct_iovs(
            allocator, io_bytes=PAGE, target_block_bytes=PAGE
        )
        self.assertEqual(direct._regions[0].block_size, PAGE)
        client.set_direct_iovs(direct)
        self.fill_file_pages(client, {0: 0x44, 1: 0x55, 2: 0x77})

        results = client.batch_read(
            [0, PAGE, 2 * PAGE], [crossing, outside, buf[2 * PAGE : 3 * PAGE]]
        )

        self.assertEqual(results, [PAGE, PAGE, PAGE])
        self.assertTrue(torch.all(crossing == 0x44))
        self.assertTrue(torch.all(outside == 0x55))
        self.assertTrue(torch.all(buf[2 * PAGE : 3 * PAGE] == 0x77))
        self.assertEqual(
            [root is client.iov_r for root in self.fake.used_iov_roots],
            [True, True, False],
        )
        direct.close()

    def test_iov_block_bytes_is_io_aligned_and_bounded_by_region(self):
        self.assertEqual(
            usrbio.iov_block_bytes(io_bytes=3, region_bytes=100, target_bytes=10), 9
        )
        self.assertEqual(
            usrbio.iov_block_bytes(io_bytes=16, region_bytes=100, target_bytes=8), 16
        )
        self.assertEqual(
            usrbio.iov_block_bytes(io_bytes=10, region_bytes=50, target_bytes=64), 0
        )


class TestHf3fsCompletionDrain(_Hf3fsTestBase):
    def test_out_of_order_completions_map_to_their_requests(self):
        client = self.make_client()
        self.fake.reverse_completions = True
        # Page 3 is past EOF of a 3.5-page file region: a short read.
        os.ftruncate(client.file, 3 * PAGE + PAGE // 2)
        self.fill_file_pages(client, {0: 0x01, 1: 0x02})
        dst = [torch.zeros(PAGE, dtype=torch.uint8) for _ in range(3)]

        results = client.batch_read([0, PAGE, 3 * PAGE], dst)

        self.assertEqual(results, [PAGE, PAGE, PAGE // 2])
        self.assertTrue(torch.all(dst[0] == 0x01))
        self.assertTrue(torch.all(dst[1] == 0x02))

    def test_timeout_waits_for_outstanding_io_and_fails_late_ones(self):
        """A batch never returns while an I/O it prepared can still touch its buffer."""
        client = self.make_client(client_timeout=0)
        self.fake.wait_budgets = [1, 0, 0, 1, 0, 1]
        dst = [torch.zeros(PAGE, dtype=torch.uint8) for _ in range(3)]

        results = client.batch_read([0, PAGE, 2 * PAGE], dst)

        self.assertEqual(self.fake.outstanding(), 0)
        self.assertEqual(results, [PAGE, 0, 0])
        self.assertEqual(self.fake.wait_budgets, [])

    def test_prepare_failure_still_drains_already_prepared_io(self):
        client = self.make_client()
        self.fake.execute_on_prepare = True
        self.fake.fail_prepare_after = 2
        self.fake.fail_submit = True
        src = [torch.full((PAGE,), i + 1, dtype=torch.uint8) for i in range(3)]

        results = client.batch_write([0, PAGE, 2 * PAGE], src)

        self.assertEqual(self.fake.outstanding(), 0)
        self.assertEqual(results, [PAGE, PAGE, 0])
        self.assertEqual(os.pread(client.file, PAGE, PAGE), b"\x02" * PAGE)

    def test_invalid_batch_raises_without_closing_shared_client(self):
        client = self.make_client(file_pages=4)
        with self.assertRaises(ValueError):
            client.batch_read([0, 4 * PAGE], [torch.zeros(PAGE, dtype=torch.uint8)] * 2)
        self.assertEqual(
            client.batch_read([0], [torch.zeros(PAGE, dtype=torch.uint8)]), [PAGE]
        )


class _FakeMetadataClient:
    def __init__(self, page_indices):
        self.page_indices = page_indices

    def initialize(self, rank, num_pages, namespace=None):
        pass

    def get_page_indices(self, rank, keys, namespace=None):
        return [self.page_indices.get(k) for k in keys]


def _make_backend(tmp_dir, page_indices, **kwargs):
    with mock.patch("signal.signal"), mock.patch("atexit.register"):
        return HiCacheHF3FS(
            rank=0,
            file_path=os.path.join(tmp_dir, "hicache.0.bin"),
            file_size=16 * PAGE,
            numjobs=2,
            bytes_per_page=PAGE,
            entries=2,
            client_timeout=1,
            dtype=torch.uint8,
            metadata_client=_FakeMetadataClient(page_indices),
            **kwargs,
        )


class TestHiCacheHF3FSBackend(_Hf3fsTestBase):
    def test_partial_metadata_miss_reads_hits_into_their_own_pages(self):
        """With a leading miss, each hit must land in its own key's destination."""
        backend = _make_backend(
            self.tmp.name, {"b": 1, "c": 2, "d": 3}, use_mock_client=True
        )
        self.addCleanup(backend.close)
        for page_index in (1, 2, 3):
            os.pwrite(
                backend.clients[0].file, bytes([page_index]) * PAGE, page_index * PAGE
            )
        values = [torch.zeros(PAGE, dtype=torch.uint8) for _ in range(4)]

        results = backend._batch_get(["a", "b", "c", "d"], values)

        self.assertEqual(results, [False, True, True, True])
        self.assertEqual(int(values[0].sum()), 0)
        for i in (1, 2, 3):
            self.assertTrue(torch.all(values[i] == i))

    def test_detach_keeps_host_pool_and_reattach_falls_back_to_staging(self):
        """Detach drops only 3FS registrations; the pool's /dev/shm name never outlives
        its first registration, so a later attach must use staging, not fail."""
        allocator = Hf3fsHostTensorAllocator()
        kv = allocator.allocate((2, 4, PAGE), dtype=torch.uint8, device="cpu")
        pool = mock.MagicMock(spec=HostKVCache)
        pool.layout = "page_first_direct"
        pool.allocator = allocator
        name = allocator.allocations[0].name
        self.assertIn(name, os.listdir("/dev/shm"))

        backend = _make_backend(self.tmp.name, {}, zero_copy_host_pool=True)
        backend.register_mem_pool_host(pool)
        direct = backend.direct_iovs
        self.assertIsNotNone(direct)
        self.assertTrue(all(c.direct_iovs is direct for c in backend.clients))
        self.assertNotIn(name, os.listdir("/dev/shm"))
        backend.close()
        backend.close()
        self.assertIsNone(backend.direct_iovs)
        kv[1, 3].fill_(7)
        self.assertTrue(torch.all(kv[1, 3] == 7))

        backend = _make_backend(self.tmp.name, {}, zero_copy_host_pool=True)
        self.addCleanup(backend.close)
        backend.register_mem_pool_host(pool)
        self.assertIsNone(backend.direct_iovs)
        self.assertEqual(self.fake.registered_names, [name])

    def test_anonymous_host_pool_falls_back_to_staging(self):
        pool = mock.MagicMock(spec=HostKVCache)
        pool.layout = "page_first"
        pool.allocator = Hf3fsHostTensorAllocator(shm_dir="/nonexistent-shm-dir")
        pool.allocator.allocate((PAGE,), dtype=torch.uint8, device="cpu")
        self.assertEqual(pool.allocator.allocations, [])
        backend = _make_backend(self.tmp.name, {}, zero_copy_host_pool=True)
        self.addCleanup(backend.close)

        backend.register_mem_pool_host(pool)

        self.assertIsNone(backend.direct_iovs)
        self.assertTrue(all(c.direct_iovs is None for c in backend.clients))
        self.assertEqual(self.fake.registered_names, [])


if __name__ == "__main__":
    unittest.main()
