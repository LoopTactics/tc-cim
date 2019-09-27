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

#include <iostream>
#include <string>

#include "tc/core/tactics/tactics_mapping_options.h"
#include "tc/core/mapping_options_cpp_printer.h"

namespace tc {

class TacticsMappingOptionsAsCpp {
 public:
  explicit TacticsMappingOptionsAsCpp(
      const TacticsMappingOptions& options_,
      size_t indent_ = 0)
      : options(options_), indent(indent_) {}
  const TacticsMappingOptions& options;
  size_t indent;
};

class TacticsMappingOptionsCppPrinter : public MappingOptionsCppPrinter {
 public:
  TacticsMappingOptionsCppPrinter(std::ostream& out, size_t ws = 0)
      : MappingOptionsCppPrinter(out, ws) {}

  ~TacticsMappingOptionsCppPrinter() = default;

  friend TacticsMappingOptionsCppPrinter& operator<<(
      TacticsMappingOptionsCppPrinter& prn,
      const TacticsMappingOptions& options);
};

TacticsMappingOptionsCppPrinter& operator<<(
    TacticsMappingOptionsCppPrinter& prn,
    const TacticsMappingOptions& tacticsOptions);

std::ostream& operator<<(std::ostream& out, const TacticsMappingOptionsAsCpp& mo);

} // namespace tc
