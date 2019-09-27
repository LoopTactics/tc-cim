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

#include <string>
#include <vector>

#include "tc/core/tactics/tactics_mapping_options.h"
#include "tc/core/tactics/tactics_mapping_options_cpp_printer.h"
#include "tc/core/tactics/tactics_rtc.h"
#include "tc/core/halide_utils.h"
#include "tc/core/tensor.h"

namespace tc {
struct WithTacticsDevice {
  WithTacticsDevice(size_t g) { }
  ~WithTacticsDevice() { }
};

/**
 * Information returned by polyhedral compilation.
 */
struct TacticsCompilationResult {
  std::string source;
  std::string specializedName;
  std::vector<long> parameters;
};

struct TacticsRuntimeInformation {
 public:
  TacticsRuntimeInformation() {}
};

struct TacticsTcExecutor;

/**
 * This type declares the dependent types and static functions needed to
 * autotune, compile and run for the CPU backend.
 */
struct TacticsBackend {
  using ExecutorType = TacticsTcExecutor;
  using MappingOptionsType = TacticsMappingOptions;
  using CompilationResultType = TacticsCompilationResult;
  using OptionsCacheProtoType = TacticsOptionsCacheProto;
  using OptionsCacheValueProtoType = TacticsOptionsCacheValueProto;
  using RTCFunctionType = TacticsRTCFunction;

  using WithDevice = WithTacticsDevice;
  using RuntimeInformation = TacticsRuntimeInformation;
  using MappingOptionsAsCpp = TacticsMappingOptionsAsCpp;
  using MappingOptionsCppPrinter = TacticsMappingOptionsCppPrinter;

  static inline std::string backendString() {
    return "loop_tactics";
  }
  static inline std::string makeDeviceFilename(const std::string& fn) {
    return fn + ".tactics";
  }

  /// Main entry point for polyhedral compilation
  static CompilationResultType compileWithTcMapper(
      const std::string& tcName,
      tc2halide::HalideComponents halideComponents,
      const std::vector<const DLConstTensor*>& inputs,
      /* TODO: in the future also pass outputs for stride and alignment */
      const MappingOptionsType& options);
};
} // namespace tc
