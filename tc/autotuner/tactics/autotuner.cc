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

#include "tc/autotuner/autotuner.h"

#include <atomic>
#include <chrono>
#include <numeric>
#include <thread>

#include <glog/stl_logging.h>

#include "tc/autotuner/utils.h"
#include "tc/core/compiler.h"
#include "tc/core/tactics/tactics_mapping_options_cpp_printer.h"
#include "tc/core/tactics/tactics_tc_executor.h"
#include "tc/core/flags.h"
#include "tc/core/scope_guard.h"
#include "tc/core/tensor.h"
#include "tc/core/tactics/tactics_backend.h"
#include "tc/core/utils/math.h"
#include "tc/lang/canonicalize.h"

namespace tc {
namespace autotune {
namespace detail {

using TacticsTuningHarness = class TuningHarness<tc::TacticsBackend>;

template <>
typename TacticsBackend::MappingOptionsType makeOptions<TacticsBackend>(
    const typename TacticsBackend::MappingOptionsType& baseMapping,
    const CandidateConfiguration& c) {
  auto options = baseMapping;
  c.configuration.applyToTacticsMappingOptions(options);
  return options;
}

template <>
TuningConfiguration makeTuningConfiguration<TacticsBackend>(
    const typename TacticsBackend::MappingOptionsType& options,
    const TuningConfiguration& configuration) {
  TuningConfiguration conf = configuration;
  conf.fromTacticsMappingOptions(options);
  return conf;
}

// This function is run on a single pre-determined GPU, in a single thread
// It takes the input/output DLTensor objects that reside on that GPU
//
// We pass the bestTimeSoFar as an option to avoid taking locks in this
// function. This trades off a bit of conservativeness for code sanity.
//
// The function returns true if pruning is possible and we can skip poorly
// performing versions early.
template <>
bool skipExecutionOrWarmup<TacticsBackend>(
    typename TacticsBackend::ExecutorType& executor,
    const std::vector<const DLTensor*>& outputs,
    const std::vector<const DLConstTensor*>& inputs,
    Duration bestTimeSoFar) {
  // 2. Perform a first run which may have one of 2 behaviors:
  //   2.a. return a very slow first execution time, we should stop
  //     early. This is akin to pruning but in this case we have run once,
  //   2.b. return a reasonable execution time, in which case we proceed with
  //     warmup.
  auto timings = executor.profile(inputs, outputs);
  // 2.a.
  constexpr size_t kCatastrophicPerfFactor = 100;
  if (bestTimeSoFar < Duration::max() and
      timings.kernelRuntime >= bestTimeSoFar * kCatastrophicPerfFactor) {
    return true;
  }
  // 2.b. during autotuning we don't want to spend too much time executing,
  // use a reduced number of iterations for warmup.
  constexpr int kReducedWarmupIterations = 2;
  for (size_t i = 1; i < kReducedWarmupIterations - 1; ++i) {
    executor.profile(inputs, outputs);
  }

  // 3. After reasonable warmup, look at the performance and prune if
  // catastrophically bad.
  constexpr int kEarlyPruneFactor = 5;
  timings = executor.profile(inputs, outputs);
  if (bestTimeSoFar < Duration::max() and
      timings.kernelRuntime >= bestTimeSoFar * kEarlyPruneFactor) {
    return true;
  }

  // 4. If we get here then the kernel is good to be benchmarked
  return false;
}

template <>
void handleDeviceRuntimeError<TacticsBackend>(
    size_t device,
    typename TacticsBackend::MappingOptionsType& options) {
}

template <>
std::vector<size_t> parseDevices<TacticsBackend>(const std::string& devices) {
  std::stringstream ss(devices);
  size_t device;
  std::vector<size_t> res;
  while (ss >> device) {
    res.push_back(device);
    if (ss.peek() == ',') {
      ss.ignore();
    }
  }
  return res;
}
} // namespace detail
} // namespace autotune
} // namespace tc
