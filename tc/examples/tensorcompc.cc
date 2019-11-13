/**
 * Copyright (c) 2017-2018, Facebook, Inc.
 * Copyright (c) 2019-present Inria
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
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cctype>

#include <gflags/gflags.h>
#include <dlpack/dlpack.h>
#include <gflags/gflags.h>

#include "tc/aten/aten.h"
#include "tc/core/check.h"
#include "tc/core/compiler.h"
#include "tc/core/tactics/tactics_backend.h"

DEFINE_string(input, "", "Name of the TC input file");
DEFINE_string(output, "-", "Name of the C output file");
DEFINE_string(input_shapes, "", "Input shapes e.g., I0:16,32,16/I1:3,19,28");
DEFINE_string(entrypoint, "", "Entrypoint to compile");
DEFINE_string(tactics_outer_tile_sizes,
	      "1",
	      "Comma-separated list of tile sizes for outer tiling");

DEFINE_string(tactics_fusion_strategy,
	      "Max",
	      "Fusion strategy [Min or Max]");

using DLTensorUPtr = std::unique_ptr<DLTensor, tc::DLTensorDeleter>;
using DLConstTensorUPtr = std::unique_ptr<DLConstTensor, tc::DLTensorDeleter>;

template <typename Backend>
inline at::Tensor makeATenTensor(at::ArrayRef<long int> sizes);

template <>
inline at::Tensor makeATenTensor<tc::TacticsBackend>(
    at::ArrayRef<long int> sizes) {
  return at::CPU(at::kFloat).rand(sizes);
}


std::vector<at::Tensor> cloneTensors(const std::vector<at::Tensor>& inputs)
{
  std::vector<at::Tensor> copies;
  copies.reserve(inputs.size());
  for (const auto& t : inputs) {
    copies.push_back(t.clone());
  }
  return copies;
}

class InputSizeParseException : public std::exception {
public:
  InputSizeParseException(size_t charPos, const std::string& msg) {
    std::stringstream ss;

    ss << "Character " << charPos << msg;

    msg_ = ss.str();
  }

  const char* what() {
    return msg_.c_str();
  }

protected:  
  std::string msg_;
};

// Parses a string of the form
// <InputName>:<size>,<size>,.../<InputName>:<size>,<size>,...
std::unordered_map<std::string, std::vector<size_t>> parseShapes(const std::string& cliparam)
{
  std::unordered_map<std::string, std::vector<size_t>> ret;
  std::vector<size_t> currSizes;
  std::string currInputName;
  size_t currSize;
  size_t pos = 0;
  char lastChar;
  
  enum mode {
     INPUT_NAME,
     SIZE_LIST
  } currMode = INPUT_NAME;
  
  for(char c: cliparam) {
    ++pos;

    if(currMode == INPUT_NAME) {
      if (isalnum(c)) {
        currInputName += c;
      } else if (c == ':') {
	if(currInputName.empty())
	  throw InputSizeParseException(pos, "Empty input name");

        currMode = SIZE_LIST;
	currSize = 0;
	currSizes.clear();
      } else if (c == '/') {
        if (currInputName.empty())
          throw InputSizeParseException(pos, "Empty input name");

        ret.emplace(std::make_pair(currInputName, std::vector<size_t>{}));
        currInputName.clear();
      } else {
	if(currInputName.empty())
	  throw InputSizeParseException(pos, "Unexpected character");
      }
    } else if(currMode == SIZE_LIST) {
      if(isdigit(c)) {
	currSize *= 10;
	currSize += c - '0'; 
      } else if(c == ',') {
	if(!isdigit(lastChar))
	  throw InputSizeParseException(pos, "Unexpected character");

	currSizes.push_back(currSize);
	currSize = 0;
      } else if(c == '/') {
	currMode = INPUT_NAME;

	if(lastChar != ':' && lastChar != ',')
	  currSizes.push_back(currSize);

	ret.emplace(std::make_pair(currInputName, currSizes));
	currSize = 0;
	currSizes.clear();
	currInputName.clear();
      }
    }

    lastChar = c;
  }

  if(currMode == SIZE_LIST) {
    if(currInputName.empty())
      throw InputSizeParseException(pos, "Empty input name");

    if(lastChar != ':' && lastChar != ',')
      currSizes.push_back(currSize);

    ret.emplace(std::make_pair(currInputName, currSizes));
  }

  return ret;
}

std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
{
  os << "[";

  for(auto& vv: v)
    os << vv << ", ";
  
  os << "]";

  return os;
}

template <typename Backend> void compile()
{
  std::unordered_map<std::string, std::vector<size_t>> cliInputSizes = parseShapes(FLAGS_input_shapes);
  
  std::ifstream ins(FLAGS_input, std::ios::in);

  if(!ins.is_open()) {
    std::cerr << "Error: Failed to open input file " << FLAGS_input << std::endl;
    throw;
  }
    
  std::string tc((std::istreambuf_iterator<char>(ins)),
		  std::istreambuf_iterator<char>());

  //  auto mappingOptions = Backend::MappingOptionsType::makeNaiveMappingOptions();
  auto mappingOptions = Backend::MappingOptionsType::makeSingleThreadMappingOptions();

  tc::CompilerOptions compilerOptions;

  std::string entryPointName = FLAGS_entrypoint;
  
  std::map<std::string, lang::TreeRef> parsedTcs = tc::detail::parse(tc);

  if(parsedTcs.size() == 0) {
    std::cerr << "No entry points found in " << FLAGS_input << std::endl;
    throw;
  }
  
  if(entryPointName == "") {
    if(parsedTcs.size() == 1) {
      entryPointName = parsedTcs.begin()->first;
    } else {
      std::cerr << "More than one entry point found in input file; "
		<< "please specify an entrypoint using --entrypoint"
		<< std::endl;
      throw;
    }
  }

  if(parsedTcs.find(entryPointName) == parsedTcs.end()) {
    std::cerr << "Entry point `" << entryPointName << "' not found"
	      << std::endl;
    throw;
  }

  mappingOptions.tile(FLAGS_tactics_outer_tile_sizes);

  mappingOptions.scheduleFusionStrategy(FLAGS_tactics_fusion_strategy);
  
  lang::TreeRef entryPoint = parsedTcs[entryPointName];

  lang::Def entryFunction(lang::Sema(compilerOptions).checkFunction(entryPoint));

  std::vector<at::Tensor> inputs;
  
  for(const auto& param: entryFunction.params()) {
    const std::string& paramName = param.ident().name();

    auto shapeIter = cliInputSizes.find(paramName);
    
    if(shapeIter == cliInputSizes.end()) {
      std::cerr << "Shape for " << paramName << " not found" << std::endl;
      throw;
    }

    if(param.tensorType().dims().size() != shapeIter->second.size()) {
      std::cerr << "Excpected number of dimensions for " << paramName << ": "
		<< param.tensorType().dims().size()
		<< ", but " << shapeIter->second.size()
		<< " dimensions specified"
		<< std::endl;
      throw;
    }

    at::ArrayRef<long int> atShape(reinterpret_cast<long int*>(shapeIter->second.data()),
				   shapeIter->second.size());

    at::Tensor t = makeATenTensor<Backend>(atShape);
    inputs.push_back(std::move(t));
  }
  
  // first parse the devices
  std::vector<size_t> devices = {0};

  std::unordered_map<size_t, std::vector<DLConstTensorUPtr>> inputsPerDevice;
  std::unordered_map<size_t, std::vector<const DLConstTensor*>> rawInputsPerDevice;

  for (auto device : devices) {
    typename Backend::WithDevice wd(device);
    auto deviceInputs = cloneTensors(inputs);
    inputsPerDevice.emplace(device, tc::aten::makeDLConstTensors(deviceInputs));
    rawInputsPerDevice.emplace(device, tc::extractRawPtrs(inputsPerDevice.at(device)));
  }

  auto halideComponents = tc2halide::translate(isl::with_exceptions::globalIslCtx(), entryPoint, compilerOptions);
  tc::detail::checkInputsCompliant(halideComponents, rawInputsPerDevice.begin()->second);
  
  typename Backend::CompilationResultType compilationResult = Backend::compileWithTcMapper(
      entryPointName,
      halideComponents,
      rawInputsPerDevice.begin()->second,
      mappingOptions);

  if(FLAGS_output == "-" || FLAGS_output == "") {
    std::cout << compilationResult.source << std::endl;
  } else {
    std::ofstream ofs(FLAGS_output, std::ios::binary);
    ofs << compilationResult.source;
  }
}

// From root, run with:
//   ./build/examples/tensordot --tuner_threads=10 --tuner_gen_pop_size=10
//   --tuner_gen_generations=3 --tuner_gen_number_elites=4
int main(int argc, char** argv) {
  try {  
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    ::google::InitGoogleLogging(argv[0]);
    tc::aten::setAtenSeed(tc::initRandomSeed(), at::Backend::CPU);
    compile<tc::TacticsBackend>();
  } catch(std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}
