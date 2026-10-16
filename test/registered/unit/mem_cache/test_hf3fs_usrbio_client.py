"""Completion ordering and buffer lifetime for the native 3FS client.

The ring and memcpy extension are replaced at the external boundary; the client
operates on real CPU tensors. No 3FS mount or compiled extension is required.
"""

import importlib.util
import threading
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _read_shm(shm, tensors):
    current = 0
    for tensor in tensors:
        dst = tensor.view(torch.uint8).reshape(-1)
        dst.copy_(shm[current : current + dst.numel()])
        current += dst.numel()


class _Ring:
    """Script native completion batches, timeouts, and transient errors."""

    def __init__(self, storage, steps):
        self.storage = storage
        self.steps = deque(steps)
        self.pending = {}
        self.prepared = 0
        self.fail_prepare_at = None
        self.fail_submit = False
        self.results = {}

    def prepare(self, iov, read, fd, offset, userdata=None):
        index = self.prepared
        if index == self.fail_prepare_at:
            raise OSError("prepare failed")
        self.pending[index] = (iov, read, offset, userdata)
        self.prepared += 1
        return self

    def submit(self):
        if self.fail_submit:
            raise OSError("submit failed")
        return self

    def wait(self, *, max_results=0, min_results=0, timeout=None):
        # Like the native binding, a wait for N results holds back earlier
        # completions until N arrive or a timeout/error interrupts the wait.
        max_results = max(max_results, min_results)
        out = []
        while len(out) < min_results:
            step = self.steps.popleft() if self.steps else list(self.pending)
            if isinstance(step, Exception):
                if out:
                    self.steps.appendleft(step)
                    break
                raise step
            if not step:
                break
            available = max_results - len(out)
            if len(step) > available:
                self.steps.appendleft(step[available:])
                step = step[:available]
            for index in step:
                iov, read, offset, userdata = self.pending.pop(index)
                result = self.results.get(index, iov.numel())
                if result > 0:
                    data = self.storage[offset : offset + result]
                    if read:
                        iov[:result].copy_(data)
                    else:
                        data.copy_(iov[:result])
                out.append(SimpleNamespace(userdata=userdata, result=result))
        return out


class TestHf3fsUsrBioClient(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Load a private module instance so replacing the import-time native
        # extension cannot affect clients imported by other tests.
        spec = importlib.util.find_spec(
            "sglang.srt.mem_cache.storage.hf3fs.hf3fs_usrbio_client"
        )
        cls.module = importlib.util.module_from_spec(spec)
        with patch(
            "sglang.srt.utils.cpp_extension_loader.load_extension_with_recovery",
            return_value=SimpleNamespace(read_shm=_read_shm),
        ):
            spec.loader.exec_module(cls.module)

    def setUp(self):
        self.client = self.module.Hf3fsUsrBioClient.__new__(
            self.module.Hf3fsUsrBioClient
        )
        self.client.entries = 4
        self.client.bytes_per_page = 16
        self.client.size = 256
        self.client.client_timeout = 1
        self.client.file = -1
        self.client.rlock = threading.RLock()
        self.client.wlock = threading.RLock()
        self.client.shm_r_tensor = torch.full((64,), 255, dtype=torch.uint8)
        self.client.shm_w_tensor = torch.zeros(64, dtype=torch.uint8)
        self.client.iov_r = self.client.shm_r_tensor
        self.client.iov_w = self.client.shm_w_tensor
        self.storage = torch.arange(256, dtype=torch.uint8)
        self.offsets = [17, 41, 73]
        # Unequal byte lengths catch fixed-page-stride and dtype-size mistakes.
        self.tensors = [
            torch.zeros(3, dtype=torch.int16),
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(5, dtype=torch.uint8),
        ]
        self.ring = _Ring(self.storage, [[2], [0, 1]])
        self.client.ior_r = self.ring
        self.copy_pending = []

        def record_copy(shm, tensors):
            self.copy_pending.append(len(self.ring.pending))
            _read_shm(shm, tensors)

        patcher = patch.object(
            self.module.hf3fs_utils, "read_shm", side_effect=record_copy
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def assert_page(self, index):
        tensor = self.tensors[index]
        data = tensor.view(torch.uint8)
        start = self.offsets[index]
        torch.testing.assert_close(data, self.storage[start : start + data.numel()])

    def test_read_copies_ready_pages_while_other_reads_are_pending(self):
        """Waiting for the whole batch loses copy/I/O overlap."""
        with self.assertNoLogs(self.module.logger, level="WARNING"):
            results = self.client.batch_read(self.offsets, self.tensors)
        self.assertEqual(results, [6, 8, 5])
        self.assertEqual(self.copy_pending, [2, 0, 0])
        for index in range(3):
            self.assert_page(index)
        self.assertFalse(self.ring.pending)

    def test_read_preserves_destinations_for_short_and_failed_completions(self):
        """Short/error pages must not expose stale bytes from staging memory."""
        self.ring.results = {0: -5, 1: 3}
        results = self.client.batch_read(self.offsets, self.tensors)
        self.assertEqual(results, [-5, 3, 5])
        for tensor in self.tensors[:2]:
            self.assertEqual(torch.count_nonzero(tensor).item(), 0)
        self.assert_page(2)
        self.assertFalse(self.ring.pending)

    def test_read_drains_after_prepare_or_submit_failure(self):
        """Published requests outlive a later preparation/notification error."""
        for failure in ("prepare", "submit"):
            with self.subTest(failure=failure):
                self.ring = _Ring(self.storage, [[0]])
                self.client.ior_r = self.ring
                if failure == "prepare":
                    self.ring.fail_prepare_at = 1
                else:
                    self.ring.fail_submit = True
                for tensor in self.tensors:
                    tensor.zero_()
                with self.assertLogs(self.module.logger, level="ERROR"):
                    results = self.client.batch_read(self.offsets, self.tensors)
                self.assertEqual(
                    results, [6, 0, 0] if failure == "prepare" else [6, 8, 5]
                )
                self.assert_page(0)
                self.assertFalse(self.ring.pending)

    def test_read_drains_through_timeout_and_wait_error(self):
        """A partial wait cannot release staging memory still owned by 3FS."""
        self.ring.steps = deque([[2], [], OSError("wait failed"), [1], [0]])
        with self.assertLogs(self.module.logger, level="WARNING"):
            results = self.client.batch_read(self.offsets, self.tensors)
        self.assertEqual(results, [6, 8, 5])
        for index in range(3):
            self.assert_page(index)
        self.assertFalse(self.ring.pending)

    def test_read_drains_remaining_pages_after_copy_failure(self):
        """A failed destination copy must not release another page's buffer."""

        def copy_after_failure(shm, tensors):
            if tensors[0] is self.tensors[2]:
                raise RuntimeError("copy failed")
            _read_shm(shm, tensors)

        with patch.object(
            self.module.hf3fs_utils,
            "read_shm",
            side_effect=copy_after_failure,
        ):
            with self.assertLogs(self.module.logger, level="ERROR"):
                results = self.client.batch_read(self.offsets, self.tensors)
        self.assertEqual(results, [6, 8, 0])
        self.assert_page(0)
        self.assert_page(1)
        self.assertEqual(torch.count_nonzero(self.tensors[2]).item(), 0)
        self.assertFalse(self.ring.pending)


if __name__ == "__main__":
    unittest.main()
