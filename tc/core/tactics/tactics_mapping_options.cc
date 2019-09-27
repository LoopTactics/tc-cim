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
#include "tc/core/tactics/tactics_mapping_options.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <type_traits>

#include "tc/proto/mapping_options.pb.h"

#include "tc/core/tactics/tactics_mapping_options_cpp_printer.h"
#include "tc/core/flags.h"
#include "tc/core/utils/string.h"

#include "tc/external/isl.h"

namespace tc {
//
// TacticsMappingOptions
//
TacticsMappingOptions::TacticsMappingOptions()
    : ownedProto_(),
      generic(*ownedProto_.mutable_generic_mapping_options())
{}

TacticsMappingOptions::TacticsMappingOptions(const TacticsMappingOptions& options)
    : ownedProto_(options.ownedProto_),
      generic(*ownedProto_.mutable_generic_mapping_options())
{}

TacticsMappingOptions::TacticsMappingOptions(const std::string& str)
    : TacticsMappingOptions() {
  generic = MappingOptionsView(*ownedProto_.mutable_generic_mapping_options());
  bool parsed = ownedProto_.ParseFromString(str);
  TC_CHECK(parsed) << "could not parse protobuf string";
}

TacticsMappingOptions& TacticsMappingOptions::operator=(
    const TacticsMappingOptions& options) {
  ownedProto_ = options.ownedProto_; // views already point to the proper place
  return *this;
}

bool TacticsMappingOptions::operator==(const TacticsMappingOptions& options) const {
  return ownedProto_.SerializeAsString() ==
      options.ownedProto_.SerializeAsString();
}

bool TacticsMappingOptions::operator!=(const TacticsMappingOptions& options) const {
  return ownedProto_.SerializeAsString() !=
      options.ownedProto_.SerializeAsString();
}

TacticsMappingOptions& TacticsMappingOptions::genericMappingOptions(
    const MappingOptions& options) {
  *(ownedProto_.mutable_generic_mapping_options()) = options.view.proto;
  return *this;
}


//
// Predefined strategies
//
TacticsMappingOptions TacticsMappingOptions::makeUnmappedMappingOptions() {
  TacticsMappingOptions mo;
  mo.genericMappingOptions(MappingOptions::makeUnmappedMappingOptions());
  return mo;
}

TacticsMappingOptions TacticsMappingOptions::makeNaiveMappingOptions() {
  return makeUnmappedMappingOptions()
    .tile(32, 32, 32)
    .unroll(1);
}

TacticsMappingOptions TacticsMappingOptions::makeSingleThreadMappingOptions() {
  return makeUnmappedMappingOptions()
      .tile(1)
      .unroll(1);
}

std::ostream& operator<<(
    std::ostream& os,
    const TacticsMappingOptions& tacticsOptions) {
  OstreamBoolalphaScope scope(os);
  tc::TacticsMappingOptionsAsCpp cpp(tacticsOptions);
  os << cpp;
  return os;
}
} // namespace tc
