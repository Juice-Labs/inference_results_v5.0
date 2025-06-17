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
from code.common.constants import Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *

ParentConfig = import_module("configs.mixtral-8x7b")
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
        'kvcache_free_gpu_mem_frac': 0.90,
        'enable_chunked_context': True,
    }

    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8',
        'effective_bits': 8.75,
        'num_calib_steps': 16,
    }

    trtllm_build_flags = {
        'tokens_per_block': 32,
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
        'max_num_tokens': 16 * 1024,
    }


class BlackwellOfflineGPUBaseConfig(OfflineGPUBaseConfig):
    precision = 'fp4'
    # vboost_slider = 1

    trtllm_runtime_flags = {
        'kvcache_free_gpu_mem_frac': 0.95,
    }

    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8',
        'effective_bits': 7.25,
        'num_score_steps': 8,
        'num_calib_steps': 64,
    }

    trtllm_build_flags = {
        'tokens_per_block': 32,
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
        'max_num_tokens': 20 * 1024,
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_144GB_aarch64x1(HopperOfflineGPUBaseConfig):
    system = KnownSystem.GH200_144GB_ARMx1
    gpu_batch_size = {'mixtral-8x7b': 4096}
    offline_expected_qps = 57
    trtllm_build_flags = {'max_num_tokens': 10 * 1024}
    trtllm_runtime_flags = {
        'max_num_tokens': 9 * 1024,
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_144GB_aarch64x2(GH200_144GB_aarch64x1):
    system = KnownSystem.GH200_144GB_ARMx2
    offline_expected_qps = GH200_144GB_aarch64x1.offline_expected_qps * 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    gpu_batch_size = {'mixtral-8x7b': 896}
    offline_expected_qps = 46
    trtllm_build_flags = {'max_num_tokens': 8192}
    trtllm_runtime_flags = {'max_num_tokens': 8192}


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2")
class H100_SXM_80GB_TP2x1(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx2
    gpu_batch_size = {'mixtral-8x7b': 4096}
    offline_expected_qps = 95


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx8(H100_SXM_80GBx1):
    system = KnownSystem.H100_SXM_80GBx8
    offline_expected_qps = 46 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx1(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GBx1
    gpu_batch_size = {'mixtral-8x7b': 3072}
    offline_expected_qps = 56
    trtllm_runtime_flags = {
        'max_num_tokens': 9 * 1024,
    }


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx1_Triton(H200_SXM_141GBx1):
    use_triton = True
    triton_num_frontends_per_model = 1
    triton_num_clients_per_frontend = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx8(H200_SXM_141GBx1):
    system = KnownSystem.H200_SXM_141GBx8
    offline_expected_qps = 56 * 8 * 0.97


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx8_Triton(H200_SXM_141GBx8):
    use_triton = True
    triton_num_frontends_per_model = 1
    triton_num_clients_per_frontend = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H200_SXM_141GBx8_MaxQ(H200_SXM_141GBx8):
    power_limit = 500
    offline_expected_qps = 41 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx1(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GB_CTSx1
    gpu_batch_size = {'mixtral-8x7b': 3072}
    offline_expected_qps = 55


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GB_CTSx8(H200_SXM_141GB_CTSx1):
    system = KnownSystem.H200_SXM_141GB_CTSx8
    offline_expected_qps = 450


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class B200_SXM_180GBx1(BlackwellOfflineGPUBaseConfig):
    system = KnownSystem.B200_SXM_180GBx1
    gpu_batch_size = {'mixtral-8x7b': 6 * 1024}
    trtllm_runtime_flags = {
        'max_num_tokens': 14 * 1024
    }
    offline_expected_qps = 110


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


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2PP2")
class GB200_NVL_186GB_ARMx4(OfflineGPUBaseConfig):
    system = KnownSystem.GB200_NVL_186GB_ARMx4
    enable_sort = False
    precision = 'fp4'

    offline_expected_qps = 120
    gpu_batch_size = {'mixtral-8x7b': 3072}
    tensor_parallelism = 1
