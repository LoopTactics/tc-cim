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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tc/core/polyhedral/scop.h"
#include "tc/core/tactics/tactics_mapping_options.h"
#include "tc/core/tensor.h"
#include "tc/external/isl.h"

namespace tc {
namespace polyhedral {
namespace detail {
class ScheduleTree;
} // namespace detail

namespace tactics {

class MatMulInfo {
  public:
    MatMulInfo() = default;
    MatMulInfo(const MatMulInfo &m) = default;

  public:
    std::string readFromA = "nullptr";
    std::string readFromB = "nullptr";
    std::string readFromC = "nullptr";
    std::string writeToC = "nullptr";

    std::string beta = "1";
    std::string alpha = "1";

    int m = -1;
    int n = -1;
    int l = -1;

    int i = -1;
    int j = -1;
    int k = -1;

    bool isAtranspose = false;
    bool isBtranspose = false;
};

class GemvInfo {
 public:
  GemvInfo() = default;
  GemvInfo(const GemvInfo& g) = default;

 public:
  std::string readFromA = "nullptr";
  std::string readFromX = "nullptr";
  std::string readFromY = "nullptr";
  std::string writeToY = "nullptr";

  std::string beta = "1";
  std::string alpha = "1";

  int m = -1;
  int n = -1;

  int i = -1;
  int j = -1;

  bool isAtranspose = false;

  // what about the type FLOAT or DOUBLE?
};

class BlasInfo {
  public:
    BlasInfo() = default;
  public:
    MatMulInfo mmi;
    GemvInfo mvi;
};
  

// Scop associated with fixed block and grid dimensions.
//
// Different branches of the schedule tree may be mapped to GPU blocks or
// threads.  The role of this class is to ensure that the number of required
// blocks and threads is consistent for the entire Scop.  It does so by
// requiring to provide grid and block configuration when constructing its
// instance.  Different parts of the schedule tree may be mapped to blocks and
// threads but the values remain those specified at construction.  If less
// blocks or threads is necessary to execute certain parts of the Scop, the
// blocks or threads dimensions will be further restricted locally in a
// specific branch of schedule tree.
//
// Two invariants must be preserved:
// 1. All paths from schedule tree root to its leaves must have exactly the
//    same number of block and thread mappings.  Code generation will fail if
//    it is not the case (TODO: automatically map to 1 thread and 1 block
//    instead).
// 2. Mapping to each block and thread must appear exactly once on each path
//    from schedule tree root to its leaves.  Mapping will fail if this
//    invariant is violated.
//
// Only const and copy accessors to the members of the original Scop are
// exposed since mapping to blocks and threads introduces schedule tree
// elements incompatible with other Scop modifications.
class MappedScop {
 private:
  MappedScop(std::unique_ptr<Scop>&& scop, uint64_t unroll_)
      : scop_(std::move(scop)), unroll(unroll_) {}

 public:
  // The MappedScop returned by this method does not satisfy the invariant
  // of having a mapping to blocks and threads.  It is up to the caller
  // to insert these mappings.
  static inline std::unique_ptr<MappedScop> makeMappedScop(
      std::unique_ptr<Scop>&& scop,
      uint64_t unroll) {
    return std::unique_ptr<MappedScop>(new MappedScop(std::move(scop), unroll));
  }

  // Prepare for sequential code
  static std::unique_ptr<MappedScop> makeWithSequentialStrategy(
      std::unique_ptr<Scop>&& scopUPtr,
      const TacticsMappingOptions& mappingOptions);

  // Fix the values of the specified parameters in the context
  // to the corresponding specified values.
  template <typename T>
  void fixParameters(const std::unordered_map<std::string, T>& sizes) {
    scop_->fixParameters(sizes);
  }

  // Insert a context node for the block and thread identifiers.
  void insertMappingContext();

  // Generate Tactics code at the current state of transformation provided a
  // name for the generated function.
  std::string codegen(const std::string& specializedName) const;

  // Accessors..
  // Const accessor to schedule of underlying Scop.
  inline const detail::ScheduleTree* schedule() const {
    return scop_->scheduleRoot();
  }
  // Reference to underlying scop, no ownership transfer intended.
  inline const Scop& scop() const {
    return *scop_;
  }
  inline Scop& scop() {
    return *scop_;
  }

 private:
  // Map the elements in "list" to successive blocks or thread identifiers,
  // with the first element mapped to identifier X.
  // Return a pointer to the updated node (below the inserted filter)
  // for call chaining purposes.
  template <typename MappingTypeId>
  detail::ScheduleTree* map(
      detail::ScheduleTree* tree,
      isl::union_pw_aff_list list);

 private:
  std::unique_ptr<Scop> scop_;

 public:
  const uint64_t unroll;
};

// Mappings from isl ids used by mark nodes to the metadata to
// generated the replacement code
struct TacticsReplacements {
  std::unordered_map<isl::id, MatMulInfo, isl::IslIdIslHash> matmul;
};

} // namespace tactics
} // namespace polyhedral
} // namespace tc
