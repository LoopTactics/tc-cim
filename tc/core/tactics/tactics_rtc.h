/**
 * Copyright (c) 2017-2018, Facebook, Inc.
 * Copyright (c) 2019-present, Inria
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

#include <memory>
#include "tc/core/utils/time.h"

namespace tc {

//
// Basic interface to expose NVRTC JIT compilation and module
// loading/unloading + API kernel launches.
//
class TacticsRTCFunction {
  TacticsRTCFunction();

 public:
  ~TacticsRTCFunction();

  static std::unique_ptr<TacticsRTCFunction> Compile(
      const std::string& name,
      const std::string& source);

  // if profile is set it returns the kernel runtime
  Duration Launch(
      // by copy because we take an address to element when calling the kernel
      // TODO: check the overhead of double indirection on kernel calls, this
      // does not look ideal for low-latency
      std::vector<long> params,
      std::vector<void*> outputs,
      std::vector<const void*> inputs,
      bool profile = false) const;

  void clear();

 private:
  std::string specializedName;
  std::vector<char> compiledSO;
};

} // namespace tc
