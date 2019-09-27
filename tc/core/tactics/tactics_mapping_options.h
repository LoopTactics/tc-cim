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

#include "tc/proto/mapping_options.pb.h"

#include <array>
#include <iostream>
#include <string>
#include <vector>

#include "tc/external/isl.h"

#include "tc/core/flags.h"
#include "tc/core/mapping_options.h"

namespace tc {
class TacticsMappingOptions {
 private:
  TacticsMappingOptions();
  static TacticsMappingOptions makeUnmappedMappingOptions();

 public:
  /// Construct a deep copy of the options.
  TacticsMappingOptions(const TacticsMappingOptions& options);
  //explicit TacticsMappingOptions(const CudaMappingOptionsProto& buf);
  TacticsMappingOptions& operator=(const TacticsMappingOptions& options);

  /// Compare with another message.
  bool operator==(const TacticsMappingOptions& options) const;
  bool operator!=(const TacticsMappingOptions& options) const;

  /// Construct from a serialized protocol buffer message.
  explicit TacticsMappingOptions(const std::string& str);

  std::string toProtobufSerializedString() const {
    return ownedProto_.SerializeAsString();
  }

  /// Set mappings
  TacticsMappingOptions& genericMappingOptions(const MappingOptions& options);
  ///@}

  /// Static constructors for predefined strategies.
  ///@{
  static TacticsMappingOptions makeNaiveMappingOptions();
  static TacticsMappingOptions makeSingleThreadMappingOptions();
  ///@}

  const TacticsMappingOptionsProto& proto() const {
    return ownedProto_;
  }

#define FORWARD_FUN(FUN_NAME)                         \
  template <typename... Args>                         \
  inline TacticsMappingOptions& FUN_NAME(Args... args) { \
    generic.FUN_NAME(args...);                        \
    return *this;                                     \
  }

  FORWARD_FUN(tile);
  FORWARD_FUN(unroll);
  FORWARD_FUN(fixParametersBeforeScheduling);
  FORWARD_FUN(tileImperfectlyNested);
  FORWARD_FUN(matchLibraryCalls);
  FORWARD_FUN(scheduleFusionStrategy);
  FORWARD_FUN(outerScheduleFusionStrategy);
  FORWARD_FUN(outerScheduleAllowSkewing);
  FORWARD_FUN(outerSchedulePositiveOrthant);
  FORWARD_FUN(intraTileScheduleFusionStrategy);
  FORWARD_FUN(intraTileScheduleAllowSkewing);
  FORWARD_FUN(intraTileSchedulePositiveOrthant);

#undef FORWARD_FUN

 private:
  TacticsMappingOptionsProto ownedProto_;

 public:
  MappingOptionsView generic;
};

std::ostream& operator<<(std::ostream& os, const TacticsMappingOptions& view);

} // namespace tc
