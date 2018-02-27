# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

import tensor_comprehensions as tc

import torch
import torch.cuda
import unittest


class TestMatmulReuse(unittest.TestCase):

    def test_matmul_reuse(self):
        lang = """
        def matmul(float(M,N) A, float(N,K) B) -> (output) {
          output(i, j) +=! A(i, kk) * B(kk, j)
        }
        """
        matmul = tc.define(lang, name="matmul")
        mat1, mat2 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        out = matmul(mat1, mat2)
        # reuse the same outputs now instad of allocating again, so we save
        # overhead of allocating storage again. Also, if the input sizes are same
        # we skip the compilation and run directly
        mat3, mat4 = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        matmul(mat3, mat4, outputs=out)


if __name__ == '__main__':
    unittest.main()
