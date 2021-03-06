/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "tc/aten/aten.h"
#include "tc/core/cpu/cpu_backend.h"

#if TC_WITH_CUDA
#include "tc/core/cuda/cuda_backend.h"
#endif
#include "tc/core/tactics/tactics_backend.h"

template <typename Backend>
inline at::Tensor makeATenTensor(at::ArrayRef<long int> sizes);

#if TC_WITH_CUDA
template <>
inline at::Tensor makeATenTensor<tc::CudaBackend>(
    at::ArrayRef<long int> sizes) {
  return at::CUDA(at::kFloat).rand(sizes);
}
#endif

template <>
inline at::Tensor makeATenTensor<tc::CpuBackend>(at::ArrayRef<long int> sizes) {
  return at::CPU(at::kFloat).rand(sizes);
}

template <>
inline at::Tensor makeATenTensor<tc::TacticsBackend>(
    at::ArrayRef<long int> sizes) {
  return at::CPU(at::kFloat).rand(sizes);
}
