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


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    min_duration = 1200000
    enable_sort = False


class HopperServerGPUBaseConfig(ServerGPUBaseConfig):
    precision = 'fp8'
    vboost_slider = 1

    trtllm_build_flags = {
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
    }

    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8'
    }

    trtllm_runtime_flags = {
        'kvcache_free_gpu_mem_frac': 0.90
    }


class BlackwellServerGPUBaseConfig(ServerGPUBaseConfig):
    precision = "fp4"
    # vboost_slider = 1

    trtllm_build_flags = {
        'gemm_plugin': 'fp4',
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx1(HopperServerGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GBx1

    gpu_batch_size = {'llama2-70b-interactive': 512}
    trtllm_build_flags = {'max_num_tokens': 256}
    trtllm_runtime_flags = {'max_num_tokens': 256}
    server_target_qps = 8.4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GBx1_HighAccuracy(H200_SXM_141GBx1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx8(H200_SXM_141GBx1):
    system = KnownSystem.H200_SXM_141GBx8
    server_target_qps = H200_SXM_141GBx1.server_target_qps * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H200_SXM_141GBx8_HighAccuracy(H200_SXM_141GBx8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class B200_SXM_180GBx1(BlackwellServerGPUBaseConfig):
    system = KnownSystem.B200_SXM_180GBx1

    gpu_batch_size = {'llama2-70b-interactive': 768}
    trtllm_build_flags = {'max_num_tokens': 768}
    trtllm_runtime_flags = {'max_num_tokens': 768}
    server_target_qps = 26


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class B200_SXM_180GBx1_HighAccuracy(B200_SXM_180GBx1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class B200_SXM_180GBx8(B200_SXM_180GBx1):
    system = KnownSystem.B200_SXM_180GBx8
    server_target_qps = 220


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class B200_SXM_180GBx8_HighAccuracy(B200_SXM_180GBx8):
    pass
