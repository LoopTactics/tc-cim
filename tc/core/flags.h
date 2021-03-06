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
#pragma once

#include <gflags/gflags.h>
#include <glog/logging.h>

namespace tc {

//
// Declare global general flags
//
DECLARE_bool(debug_lang);
DECLARE_bool(debug_halide);
DECLARE_bool(debug_tc_mapper);
DECLARE_bool(debug_cuda);
DECLARE_bool(debug_tuner);
DECLARE_bool(dump_cuda);
DECLARE_bool(dump_ptx);
DECLARE_bool(dump_tactics);

// ptx generation
DECLARE_string(cuda_compiler);
DECLARE_string(llvm_flags);
DECLARE_string(nvcc_flags);

// llvm codegen
DECLARE_bool(llvm_dump_before_opt);
DECLARE_bool(llvm_dump_after_opt);

// Used in benchmarking and autotuning
DECLARE_uint32(benchmark_warmup);
DECLARE_uint32(benchmark_iterations);

// Used in autotuning
DECLARE_uint32(tuner_max_unroll_size);
DECLARE_uint32(tuner_gen_pop_size);
DECLARE_uint32(tuner_gen_crossover_rate);
DECLARE_uint32(tuner_gen_mutation_rate);
DECLARE_uint32(tuner_gen_generations);
DECLARE_uint32(tuner_gen_number_elites);
DECLARE_uint32(tuner_threads);
DECLARE_string(tuner_devices);
DECLARE_bool(tuner_print_best);
DECLARE_string(tuner_rng_restore);
DECLARE_bool(tuner_gen_restore_from_proto);
DECLARE_uint32(tuner_gen_restore_number);
DECLARE_bool(tuner_gen_log_generations);
DECLARE_uint64(tuner_min_launch_total_threads);
DECLARE_uint32(tuner_save_best_candidates_count);

// Loop Tactics Flags
DECLARE_bool(generate_tactics_main);

// Disable Loop Tactics
DECLARE_bool(disable_tactics);

// Disable generation of a Loop Tactics autotuner entry point
DECLARE_bool(disable_tactics_entrypoint);

// Misc
DECLARE_int64(random_seed);
DECLARE_bool(schedule_tree_verbose_validation);

// random seed setting for reproducibility and debugging purposes
uint64_t initRandomSeed();
const uint64_t& randomSeed();
} // namespace tc
