import unittest

import torch
import torch_npu

import mmq_npu


class TestAdd(unittest.TestCase):
    @torch.no_grad()
    def test_add(self):
        for size in [(3, 4), (73, 234), (9123, 3985)]:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                x = torch.randn(*size, dtype=dtype)
                y = torch.randn(*size, dtype=dtype)

                out_cpu = x + y

                x = x.npu()
                y = y.npu()
                out_npu = mmq_npu.add(x, y)

                out_npu = out_npu.cpu()

                self.assertTrue(torch.allclose(out_cpu, out_npu))
