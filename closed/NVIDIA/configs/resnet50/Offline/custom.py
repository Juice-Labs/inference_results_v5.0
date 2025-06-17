# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A4000X2(OfflineGPUBaseConfig):
    system = KnownSystem.a4000x2
    gpu_batch_size = {'resnet50': 256}
    offline_expected_qps = 40000
    gpu_inference_streams = 1
    gpu_copy_streams = 2
