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

#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include "tc/core/check.h"
#include "tc/tc_config.h"
#include "tc/core/tactics/tactics_rtc.h"
#include "tc/core/flags.h"

#include "tc/core/scope_guard.h"

extern "C" {
#include <dlfcn.h>
}

namespace tc {

TacticsRTCFunction::TacticsRTCFunction() {}
TacticsRTCFunction::~TacticsRTCFunction() {}
void TacticsRTCFunction::clear() {}

namespace {
static void checkedSystemCall(
    const std::string& cmd,
    const std::vector<std::string>& args) {
  std::stringstream command;
  command << cmd << " ";
  for (const auto& s : args) {
    command << s << " ";
  }
  TC_CHECK_EQ(std::system(command.str().c_str()), 0) << command.str();
}

static std::vector<char> llvmCompileSO(
    const std::string& name,
    const std::string& source) {
  char pat[] = "/tmp/tacticsXXXXXX.c";
  int fd = mkstemps(pat, 2);
  TC_CHECK_GE(fd, 0) << "mkstemps() failed: " << strerror(errno);
  std::string inputFileName(pat);

  std::ofstream ostream(inputFileName, std::ios::binary);
  ostream << source;
  
  std::string outputFileName = inputFileName + ".so";

  tc::ScopeGuard sgi([&]() {
    close(fd);
    std::remove(inputFileName.c_str());
  });

  tc::ScopeGuard sgo([&]() {
    std::remove(outputFileName.c_str());
  });

  
  // Compile
  checkedSystemCall(
      std::string(TC_STRINGIFY(TC_LLVM_BIN_DIR)) + "/clang",
      {
       tc::FLAGS_llvm_flags,
       "-std=c99",
       "-fPIC",
       "-ffast-math",
       "-shared",
       "-o " + outputFileName,
       inputFileName});

  std::ifstream stream(outputFileName, std::ios::binary);
  
  return std::vector<char>(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
}
} // namespace

std::unique_ptr<TacticsRTCFunction> TacticsRTCFunction::Compile(
    const std::string& name,
    const std::string& source) {
  std::unique_ptr<TacticsRTCFunction> res(new TacticsRTCFunction());
  res->specializedName = name;
  res->compiledSO = llvmCompileSO(name, source);

  return res;
}

Duration TacticsRTCFunction::Launch(
    std::vector<long> params,
    std::vector<void*> outputs,
    std::vector<const void*> inputs,
    bool profile) const {
  void* so_handle = NULL;
  void* entrypoint_fun = NULL;

  //  std::cout << "Launch called" << std::endl;

  char pat[] = "/tmp/tacticsXXXXXX.so";
  int fd = mkstemps(pat, 3);
  TC_CHECK_GE(fd, 0) << "mkstemps() failed: " << strerror(errno);
  std::string soFileName(pat);

  tc::ScopeGuard sgi([&]() {
    close(fd);
    std::remove(soFileName.c_str());
  });

  std::ofstream soos(soFileName, std::ios::binary);

  std::copy(compiledSO.begin(), compiledSO.end(), std::ostream_iterator<char>(soos));
  soos.flush();
    
  so_handle = dlopen(soFileName.c_str(), RTLD_NOW);
  TC_CHECK_NE(so_handle, (void*)0) << "Could no open SO file " << soFileName;

  tc::ScopeGuard sgdl([&]() {
  			// std::cout << "Closing handle" << std::endl;
  			dlclose(so_handle);
  		      });

  // std::cout << "Loading kernel from " << soFileName << std::endl;

  entrypoint_fun = dlsym(so_handle, "tactics_entrypoint");
  TC_CHECK_NE(entrypoint_fun, (void*)0) <<
    "Could not find symbol tactics_entrypoint in " << soFileName;

  constexpr size_t kNumMaxParameters = 100;
  std::array<void*, kNumMaxParameters> args_voidp{0};
  TC_CHECK_GE(
      kNumMaxParameters, params.size() + outputs.size() + inputs.size());
  size_t ind = 0;
  for (auto& p : params) {
    args_voidp[ind++] = &p;
  }
  for (auto& o : outputs) {
    args_voidp[ind++] = &o;
  }
  for (auto& i : inputs) {
    args_voidp[ind++] = static_cast<void*>(&i);
  }

  void (*entrypoint)(void** args, size_t num_args);

  entrypoint = reinterpret_cast<void (*)(void**, size_t)>(entrypoint_fun);

  auto start_time = std::chrono::system_clock::now();
  
  entrypoint(args_voidp.data(), ind);

  auto end_time = std::chrono::system_clock::now();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  return Duration::fromMicroSeconds(us.count());
}
} // namespace tc
