# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.insert(0, os.getcwd())

from importlib import import_module
from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *

ParentConfig = import_module("configs.llama2-70b-interactive")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    min_duration = 2400000
    enable_sort = False

    trtllm_runtime_flags = {
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
    }


class HopperOfflineGPUBaseConfig(OfflineGPUBaseConfig):
    precision = "fp8"
    vboost_slider = 1

    trtllm_runtime_flags = {
        'kvcache_free_gpu_mem_frac': 0.95,
        'enable_chunked_context': True,
    }

    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8'
    }

    trtllm_build_flags = {
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
    }

    trtllm_runtime_flags = {
        'kvcache_free_gpu_mem_frac': 0.90
    }


class BlackwellOfflineGPUBaseConfig(OfflineGPUBaseConfig):
    precision = 'fp4'
    # vboost_slider = 1

    trtllm_runtime_flags = {
        'kvcache_free_gpu_mem_frac': 0.95,
        'enable_chunked_context': True,
    }

    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8'
    }

    trtllm_build_flags = {
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx1(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GBx1

    gpu_batch_size = {'llama2-70b-interactive': 2048}
    trtllm_build_flags = {
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}
    offline_expected_qps = 14.4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GBx1_HighAccuracy(H200_SXM_141GBx1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx8(H200_SXM_141GBx1):
    system = KnownSystem.H200_SXM_141GBx8
    offline_expected_qps = H200_SXM_141GBx1.offline_expected_qps * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GBx8_HighAccuracy(H200_SXM_141GBx8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class B200_SXM_180GBx1(BlackwellOfflineGPUBaseConfig):
    system = KnownSystem.B200_SXM_180GBx1

    # vboost_slider = 1
    offline_expected_qps = 47
    gpu_batch_size = {'llama2-70b-interactive': 2048}
    trtllm_build_flags = {'max_num_tokens': 3584}
    trtllm_runtime_flags = {
        'max_num_tokens': 3584,
        'kvcache_free_gpu_mem_frac': 0.95,
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class B200_SXM_180GBx1_HighAccuracy(B200_SXM_180GBx1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class B200_SXM_180GBx8(B200_SXM_180GBx1):
    system = KnownSystem.B200_SXM_180GBx8
    offline_expected_qps = B200_SXM_180GBx1.offline_expected_qps * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class B200_SXM_180GBx8_HighAccuracy(B200_SXM_180GBx8):
    pass
