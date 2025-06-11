# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A4000X2(ServerGPUBaseConfig):
    system = KnownSystem.a4000x2

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: Dict = {}
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    bot_mlp_precision: str = ''
    cache_file: str = ''
    check_contiguity: bool = False
    complete_threads: int = 0
    embedding_weights_on_gpu_part: float = 0.0
    embeddings_path: str = ''
    embeddings_precision: str = ''
    engine_dir: str = ''
    final_linear_precision: str = ''
    gpu_num_bundles: int = 0
    interaction_op_precision: str = ''
    max_pairs_per_staging_thread: int = 0
    num_staging_batches: int = 0
    num_staging_threads: int = 0
    qsl_numa_override: str = ''
    sample_partition_path: str = ''
    schedule_rng_seed: int = 0
    server_num_issue_query_threads: int = 0
    server_target_latency_ns: int = 0
    server_target_latency_percentile: float = 0.0
    server_target_qps: float = 0.0
    server_target_qps_adj_factor: float = 0.0
    top_mlp_precision: str = ''
    use_batcher_thread_per_device: bool = False
    vboost_slider: int = 0
    warmup_duration: float = 0.0
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A4000X2_HighAccuracy(A4000X2):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A4000X2_Triton(A4000X2):
    use_triton = True

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: Dict = {}
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    batch_triton_requests: bool = False
    bot_mlp_precision: str = ''
    cache_file: str = ''
    check_contiguity: bool = False
    complete_threads: int = 0
    embedding_weights_on_gpu_part: float = 0.0
    embeddings_path: str = ''
    embeddings_precision: str = ''
    engine_dir: str = ''
    final_linear_precision: str = ''
    gather_kernel_buffer_threshold: int = 0
    gpu_num_bundles: int = 0
    interaction_op_precision: str = ''
    max_pairs_per_staging_thread: int = 0
    max_queue_delay_usec: int = 0
    num_concurrent_batchers: int = 0
    num_concurrent_issuers: int = 0
    num_staging_batches: int = 0
    num_staging_threads: int = 0
    output_pinned_memory: bool = False
    qsl_numa_override: str = ''
    sample_partition_path: str = ''
    schedule_rng_seed: int = 0
    server_num_issue_query_threads: int = 0
    server_target_latency_ns: int = 0
    server_target_latency_percentile: float = 0.0
    server_target_qps: float = 0.0
    server_target_qps_adj_factor: float = 0.0
    top_mlp_precision: str = ''
    triton_grpc_ports: str = ''
    triton_num_clients_per_frontend: int = 0
    triton_num_frontends_per_model: int = 0
    triton_num_servers: int = 0
    triton_skip_server_spawn: bool = False
    triton_verbose_frontend: bool = False
    use_batcher_thread_per_device: bool = False
    use_concurrent_harness: bool = False
    vboost_slider: int = 0
    warmup_duration: float = 0.0
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A4000X2_HighAccuracy_Triton(A4000X2_HighAccuracy):
    use_triton = True


# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A4000X2(ServerGPUBaseConfig):
    system = KnownSystem.a4000x2

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: Dict = {}
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    bot_mlp_precision: str = ''
    cache_file: str = ''
    check_contiguity: bool = False
    complete_threads: int = 0
    embedding_weights_on_gpu_part: float = 0.0
    embeddings_path: str = ''
    embeddings_precision: str = ''
    engine_dir: str = ''
    final_linear_precision: str = ''
    gpu_num_bundles: int = 0
    interaction_op_precision: str = ''
    max_pairs_per_staging_thread: int = 0
    num_staging_batches: int = 0
    num_staging_threads: int = 0
    qsl_numa_override: str = ''
    sample_partition_path: str = ''
    schedule_rng_seed: int = 0
    server_num_issue_query_threads: int = 0
    server_target_latency_ns: int = 0
    server_target_latency_percentile: float = 0.0
    server_target_qps: float = 0.0
    server_target_qps_adj_factor: float = 0.0
    top_mlp_precision: str = ''
    use_batcher_thread_per_device: bool = False
    vboost_slider: int = 0
    warmup_duration: float = 0.0
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A4000X2_HighAccuracy(A4000X2):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A4000X2_Triton(A4000X2):
    use_triton = True

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: Dict = {}
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    batch_triton_requests: bool = False
    bot_mlp_precision: str = ''
    cache_file: str = ''
    check_contiguity: bool = False
    complete_threads: int = 0
    embedding_weights_on_gpu_part: float = 0.0
    embeddings_path: str = ''
    embeddings_precision: str = ''
    engine_dir: str = ''
    final_linear_precision: str = ''
    gather_kernel_buffer_threshold: int = 0
    gpu_num_bundles: int = 0
    interaction_op_precision: str = ''
    max_pairs_per_staging_thread: int = 0
    max_queue_delay_usec: int = 0
    num_concurrent_batchers: int = 0
    num_concurrent_issuers: int = 0
    num_staging_batches: int = 0
    num_staging_threads: int = 0
    output_pinned_memory: bool = False
    qsl_numa_override: str = ''
    sample_partition_path: str = ''
    schedule_rng_seed: int = 0
    server_num_issue_query_threads: int = 0
    server_target_latency_ns: int = 0
    server_target_latency_percentile: float = 0.0
    server_target_qps: float = 0.0
    server_target_qps_adj_factor: float = 0.0
    top_mlp_precision: str = ''
    triton_grpc_ports: str = ''
    triton_num_clients_per_frontend: int = 0
    triton_num_frontends_per_model: int = 0
    triton_num_servers: int = 0
    triton_skip_server_spawn: bool = False
    triton_verbose_frontend: bool = False
    use_batcher_thread_per_device: bool = False
    use_concurrent_harness: bool = False
    vboost_slider: int = 0
    warmup_duration: float = 0.0
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A4000X2_HighAccuracy_Triton(A4000X2_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A4000X2(ServerGPUBaseConfig):
    system = KnownSystem.a4000x2

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: Dict = {}
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    bot_mlp_precision: str = ''
    cache_file: str = ''
    check_contiguity: bool = False
    complete_threads: int = 0
    embedding_weights_on_gpu_part: float = 0.0
    embeddings_path: str = ''
    embeddings_precision: str = ''
    engine_dir: str = ''
    final_linear_precision: str = ''
    gpu_num_bundles: int = 0
    interaction_op_precision: str = ''
    max_pairs_per_staging_thread: int = 0
    num_staging_batches: int = 0
    num_staging_threads: int = 0
    qsl_numa_override: str = ''
    sample_partition_path: str = ''
    schedule_rng_seed: int = 0
    server_num_issue_query_threads: int = 0
    server_target_latency_ns: int = 0
    server_target_latency_percentile: float = 0.0
    server_target_qps: float = 0.0
    server_target_qps_adj_factor: float = 0.0
    top_mlp_precision: str = ''
    use_batcher_thread_per_device: bool = False
    vboost_slider: int = 0
    warmup_duration: float = 0.0
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A4000X2_HighAccuracy(A4000X2):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A4000X2_Triton(A4000X2):
    use_triton = True

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: Dict = {}
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    batch_triton_requests: bool = False
    bot_mlp_precision: str = ''
    cache_file: str = ''
    check_contiguity: bool = False
    complete_threads: int = 0
    embedding_weights_on_gpu_part: float = 0.0
    embeddings_path: str = ''
    embeddings_precision: str = ''
    engine_dir: str = ''
    final_linear_precision: str = ''
    gather_kernel_buffer_threshold: int = 0
    gpu_num_bundles: int = 0
    interaction_op_precision: str = ''
    max_pairs_per_staging_thread: int = 0
    max_queue_delay_usec: int = 0
    num_concurrent_batchers: int = 0
    num_concurrent_issuers: int = 0
    num_staging_batches: int = 0
    num_staging_threads: int = 0
    output_pinned_memory: bool = False
    qsl_numa_override: str = ''
    sample_partition_path: str = ''
    schedule_rng_seed: int = 0
    server_num_issue_query_threads: int = 0
    server_target_latency_ns: int = 0
    server_target_latency_percentile: float = 0.0
    server_target_qps: float = 0.0
    server_target_qps_adj_factor: float = 0.0
    top_mlp_precision: str = ''
    triton_grpc_ports: str = ''
    triton_num_clients_per_frontend: int = 0
    triton_num_frontends_per_model: int = 0
    triton_num_servers: int = 0
    triton_skip_server_spawn: bool = False
    triton_verbose_frontend: bool = False
    use_batcher_thread_per_device: bool = False
    use_concurrent_harness: bool = False
    vboost_slider: int = 0
    warmup_duration: float = 0.0
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A4000X2_HighAccuracy_Triton(A4000X2_HighAccuracy):
    use_triton = True


