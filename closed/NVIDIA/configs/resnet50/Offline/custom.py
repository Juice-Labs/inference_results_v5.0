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

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A4000X2_Triton(A4000X2):
    use_triton = True

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: {'resnet50': 256}
    # map_path: str = ''
    # tensor_path: str = ''

    # Optional fields:
    # active_sms: int = 0
    # assume_contiguous: bool = False
    # batch_triton_requests: bool = False
    # cache_file: str = ''
    # complete_threads: int = 0
    # disable_beta1_smallk: bool = False
    # engine_dir: str = ''
    # gather_kernel_buffer_threshold: int = 0
    # max_queue_delay_usec: int = 0
    # num_concurrent_batchers: int = 0
    # num_concurrent_issuers: int = 0
    # offline_expected_qps: float = 0.0
    # output_pinned_memory: bool = False
    # triton_grpc_ports: str = ''
    # triton_num_clients_per_frontend: int = 0
    # triton_num_frontends_per_model: int = 0
    # triton_num_servers: int = 0
    # triton_skip_server_spawn: bool = False
    # triton_verbose_frontend: bool = False
    # use_batcher_thread_per_device: bool = False
    # use_concurrent_harness: bool = False
    # use_cuda_thread_per_device: bool = False
    # use_deque_limit: bool = False
    # use_same_context: bool = False
    # vboost_slider: int = 0
    # warmup_duration: float = 0.0
    # workspace_size: int = 0
