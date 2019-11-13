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
#include "tc/core/polyhedral/tactics/mapped_scop.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "tc/core/check.h"
#include "tc/core/cuda/cuda_libraries.h"
#include "tc/core/flags.h"
#include "tc/core/functional.h"
#include "tc/core/gpu.h"
#include "tc/core/polyhedral/cuda/mapping_types.h"
#include "tc/core/polyhedral/exceptions.h"
#include "tc/core/polyhedral/schedule_transforms.h"
#include "tc/core/polyhedral/schedule_tree_matcher.h"
#include "tc/core/polyhedral/schedule_utils.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/core/polyhedral/separation.h"
#include "tc/core/polyhedral/tactics/codegen.h"
#include "tc/core/polyhedral/unroll.h"
#include "tc/core/scope_guard.h"

#include <glog/logging.h>

using tc::polyhedral::detail::ScheduleTree;
using tc::polyhedral::detail::ScheduleTreeBand;
using tc::polyhedral::detail::ScheduleTreeContext;
using tc::polyhedral::detail::ScheduleTreeSequence;

namespace tc {
namespace polyhedral {
namespace tactics {

namespace {

template <typename ExceptionType>
inline void throwIfHasPattern(
    ScheduleTreeMatcher matcher,
    const ScheduleTree* root) {
  auto candidates = match(matcher, root);
  if (candidates.size() > 0) {
    std::stringstream ss;
    ss << "Found bad pattern:\n" << *candidates[0] << "\nin:\n" << *root;
    LOG(ERROR) << ss.str();
    throw ExceptionType(ss.str());
  }
}

void validate(const ScheduleTree* root) {
  throwIfHasPattern<EmptyFilterException>(
      filter(
          [](isl::union_set uset) { return !uset || uset.is_empty(); }, any()),
      root);
  throwIfHasPattern<EmptyMappingException>(
      mapping_filter(
          [](isl::union_set uset) { return !uset || uset.is_empty(); }, any()),
      root);
}
} // namespace

namespace {
// Specialize a MappedScop with respect to its context.
// The context is explicitly inserted as a specialization context in
// the cloned underlying scop.
std::unique_ptr<MappedScop> makeSpecializedMappedScop(
    const MappedScop& mappedScop) {
  auto scop = Scop::makeScop(mappedScop.scop());

  // In this particular specialized Scop, we can add a context just below root.
  // Context nodes in the schedule tree use _set_ spaces rather than _parameter_
  // spaces because they may depend on outer schedule dimensions.  In this
  // particular case, the "root is domain" invariant guarantees there are no
  // outer schedule dimensions, so the space of a parameter context code is that
  // of a zero-dimensional space.
  auto root = scop->scheduleRoot();
  updateTopLevelContext(root, scop->context().from_params());

  auto res = MappedScop::makeMappedScop(std::move(scop), mappedScop.unroll);

  return res;
}
} // namespace

// Before generating code, make a copy of the scop and insert
// the context of the original scop as top-level
// context node in schedule tree.
std::string MappedScop::codegen(const std::string& specializedName) const {
  validate(schedule());

  auto mappedScopForCodegen = makeSpecializedMappedScop(*this);

  std::stringstream code;

  code << "#include <stdarg.h>" << std::endl
       << "#include <stdio.h>" << std::endl
       << "#include <stdlib.h>" << std::endl
       << std::endl;

  code << code::c::minmax << std::endl;

  if (mappedScopForCodegen->scop().treeSyncUpdateMap.size() != 0) {
    code << code::cuda::common;
  }
  code << emitTacticsKernel(specializedName, *mappedScopForCodegen)
       << std::endl;

  if (!FLAGS_disable_tactics_entrypoint) {
    code << emitTacticsEntryPoint(specializedName, *mappedScopForCodegen)
         << std::endl;
  }

  if (FLAGS_generate_tactics_main) {
    code << emitTacticsMain(specializedName, *mappedScopForCodegen)
         << std::endl;
  }

  return code.str();
}

std::unique_ptr<MappedScop> MappedScop::makeWithSequentialStrategy(
    std::unique_ptr<Scop>&& scopUPtr,
    const TacticsMappingOptions& mappingOptions) {
  using namespace polyhedral::detail;

  const auto& generic = mappingOptions.generic;

  // 1 block, 1 thread, no readonly cache
  auto mappedScop = std::unique_ptr<MappedScop>(
      new MappedScop(std::move(scopUPtr), generic.proto.unroll()));
  auto& scop = mappedScop->scop_;

  // 1a. Optionally specialize before scheduling...
  if (generic.proto.fix_parameters_before_scheduling()) {
    scop->specializeToContext();
  }

  // 2. Schedule
  scop = Scop::makeScheduled(*scop, generic.outerScheduleOptions);

  // 3. Tile
  TC_CHECK_LT(0u, generic.tiling.size())
      << "Must pass tile vector with >= 1 tile sizes";
  auto outerBand = scop->tileOuterBand(generic.tiling);

  // 4. Optionally reschedule if point loops need a different strategy than
  // tile loops
  if (generic.outerScheduleOptions != generic.intraTileScheduleOptions) {
    scop->reschedule(outerBand->child({0}), generic.intraTileScheduleOptions);
    LOG_IF(INFO, FLAGS_debug_tc_mapper)
        << "After intra-tile rescheduling:" << std::endl
        << *mappedScop->schedule();
  }

  // 1b. ...or after rescheduling
  if (!generic.proto.fix_parameters_before_scheduling()) {
    scop->specializeToContext();
  }

  return mappedScop;
}

} // namespace tactics
} // namespace polyhedral
} // namespace tc
