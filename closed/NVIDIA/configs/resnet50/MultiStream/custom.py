# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A4000X2(MultiStreamGPUBaseConfig):
    system = KnownSystem.a4000x2
    multi_stream_expected_latency_ns = 1809322 # 3618644 #7973184
