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
#include "tc/core/tactics/tactics_mapping_options_cpp_printer.h"

#include <sstream>

namespace tc {

TacticsMappingOptionsCppPrinter& operator<<(
    TacticsMappingOptionsCppPrinter& prn,
    const TacticsMappingOptions& tacticsOptions) {
  prn.printString("tc::TacticsMappingOptions::makeNaiveMappingOptions()");
  prn.print(tacticsOptions.generic);
  prn.endStmt();
  return prn;
}

std::ostream& operator<<(std::ostream& out, const TacticsMappingOptionsAsCpp& mo) {
  auto prn = TacticsMappingOptionsCppPrinter(out, mo.indent);
  prn << mo.options;
  return out;
}

} // namespace tc
