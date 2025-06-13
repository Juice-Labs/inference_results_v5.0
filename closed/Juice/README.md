# Juice MLPerf Inference v5.0 Benchmarks

This directory contains the NVIDIA-optimized implementations of [MLPerf](https://mlcommons.org/en/) Inference Benchmarks configured to run using Juice software to attach GPUs dynamically over the network.  The changes amount to installing and using Juice to attach GPUs in the Docker container when running the benchmarks.

Running a benchmark generally involves four steps:

1. Clone [Juice-Labs/inference_results_v5.0](https://github.com/Juice-Labs/inference_results_v5.0)
2. Copy `juice-gpu-linux.tar.gz` into `closed/Juice/docker/common`
3. Build and run the Docker container from `closed/Juice`
4. Obtain and preprocess the benchmark dataset
5. Run the benchmark

Building and running the Docker container is the same for the NVIDIA-native and Juice benchmarks.  Run commands from `closed/Juice` instead of `closed/NVIDIA`.  Follow the general instructions in [README.md](https://github.com/Juice-Labs/inference_results_v5.0/tree/charles/wip/closed/NVIDIA) up to adding a custom system and running the benchmark.

Obtaining and preprocessing datasets varies by benchmark -- some datasets must be manually downloaded and all benchmarks have different preprocessing requirements.  See the benchmark-specific documentation in, e.g. [resnet50/tensorrt/README.md](https://github.com/Juice-Labs/inference_results_v5.0/tree/charles/wip/closed/Juice) for details.

Launch the final two steps, to add a custom system and runn the benchmark, with `juice run` to attach a Juice GPU dynamically.  For example adding a custom system and running the Resnet50/Offline benchmark with Juice:

~~~sh
#...
juice login
juice run python3 -m scripts.custom_systems.add_custom_system
juice run make run RUN_ARGS="--benchmarks resnet50 --scenarios offline"
~~~

The following commands provide an overview of all the commands required to run the Resnet50 benchmark with Juice.  There are several caveats in doing this -- the ImageNet dataset needs to be downloaded manually before-hand and adding a custom system requires some editing before the benchmark will run.  See the [following section](#running-the-mlperf-resnet50-benchmark-with-juice) for details:

~~~sh
# Specifies the path to datasets
export MLPERF_SCRATCH_PATH=<path-to-scratch-space>

# Build and run a Docker container that is used to run subsequent steps
make prebuild

# Build custom tools and frameworks used by benchmarks
make build

# Preprocess the previously manually download ImageNet dataset
BENCHMARKS=resnet50 make download_data
BENCHMARKS=resnet50 make preprocess_data
BENCHMARKS=resnet50 make download_model

# Run the benchmark
juice login
juice run python3 -m scripts.custom_systems.add_custom_system
juice run make run RUN_ARGS="--benchmarks=resnet50 --scenarios=offline"
~~~

## Running the MLPerf Resnet50 Benchmark with Juice

### Download ImageNet Dataset

The benchmarks store datasets in the directory specified by the `MLPERF_SCRATCH_PATH` environment variable.  The directory specified by this variable is mounted into the Docker container and is the source of datasets when preprocessing data and when running benchmarks.  Extract any manually downloaded datasets here.  Export the environment variable before running any of the Make scripts below.

Download the [ImageNet 2012 validation dataset](http://www.image-net.org/challenges/LSVRC/2012/) from [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data) and extract it to `$MLPERF_SCRATCH_PATH/data/imagenet`.

### Build and Run the Docker Container

Launch `make prebuild` from the `closed/Juice` directory to build and run the Docker container for preprocessing datasets and running the benchmarks:

~~~sh
export MLPERF_SCRATCH_PATH=<path-to-scratch-space>
make prebuild
~~~

All other commands, e.g. to download data and models, preprocess data, and run benchmarks are launched from the shell within the Docker container launched by `make prebuild`.

Build benchmarking components:

~~~sh
make build
~~~

### Download and Preprocess the Dataset

Preprocess the ImageNet dataset and download the model by running the following from inside the Docker container.

~~~sh
BENCHMARKS=resnet50 make download_data
BENCHMARKS=resnet50 make preprocess_data
BENCHMARKS=resnet50 make download_model
~~~

For the Resnet50 benchmark the `download_data` step validates that the ImageNet dataset is extracted correctly, it doesn't download anything itself.  The ImageNet dataset needs to be download manually.  Most other datasets have datasets that are downloaded automatically when `make download_data` is run.

### Add a Custom System

Add a custom system using a Juice GPU.  This only needs to be done once:

~~~sh
juice login
juice run python3 -m scripts.custom_systems.add_custom_system
~~~

Adding a custom system adds placeholder configurations in `.../code/{benchmark}/{scenario}/custom.py` for each benchmark and scenario and the custom system itself in `.../code/common/systems/custom_list.json`.  The settings for running on the custom system, e.g. batch size for offline scenarios and latency bounds for single- and multi-stream scenarios, should be changed for optimal performance.  You will likely need to experiment with different settings to get good results -- starting from the settings from the closest equivalent NVIDIA system is a reasonable place to start.

Unfortunately these placeholder configurations won't allow running the benchmarks right away.  There are several problems that need to be fixed first:

- Change typed Python syntax like ` gpu_batch_size: Dict = {}` to `gpu_batch_size = {'resnet50': 256}`.  The Python version installed in the Docker container doesn't recognize this syntax.

- Comment out the lines that set `cache_file`, `engine_dir`, `map_path`, and `tensor_path` to empty strings.  Benchmarks fail to run when these paths are set to empty strings as files fail to be read or written.

- Add `"host_memory"` to the `_match_ignore_fields` field in `.../code/common/systems/custom_list.json`.  Available host memory is used to match the system but often changes between adding the custom system and running the benchmark.

These are the contents of the custom systems `juice_a4000x2` and `a4000x2` used to generate the Juice Resnet50 benchmark results:

~~~py
# .../Juice/configs/resnet50/Offline/custom.py
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class juice_a4000x2(OfflineGPUBaseConfig):
    system = KnownSystem.juice_a4000x2
    gpu_batch_size = {'resnet50': 256}
    offline_expected_qps = 40000
    gpu_inference_streams = 1
    gpu_copy_streams = 2

# .../Juice/configs/resnet50/SingleStream/custom.py
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class juice_a4000x2(SingleStreamGPUBaseConfig):
    system = KnownSystem.juice_a4000x2
    single_stream_expected_latency_ns = 996648

# .../Juice/configs/resnet50/MultiStream/custom.py
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class juice_a4000x2(MultiStreamGPUBaseConfig):
    system = KnownSystem.juice_a4000x2
    multi_stream_expected_latency_ns = 3618644
~~~

These are the contents of the custom systems `juice_a4000x2` and `a4000x2` used to generate the NVIDIA-native Resnet50 benchmark results.  These are largely the same as the Juice versions but note the lower expected latency for the multi-stream scenario:

~~~py
# .../NVIDIA/configs/resnet50/Offline/custom.py
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class a4000x2(OfflineGPUBaseConfig):
    system = KnownSystem.a4000x2
    gpu_batch_size = {'resnet50': 256}
    offline_expected_qps = 40000
    gpu_inference_streams = 1
    gpu_copy_streams = 2

# .../NVIDIA/configs/resnet50/SingleStream/custom.py
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class a4000x2(SingleStreamGPUBaseConfig):
    system = KnownSystem.a4000x2
    single_stream_expected_latency_ns = 996648

# .../NVIDIA/configs/resnet50/MultiStream/custom.py
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class a4000x2(MultiStreamGPUBaseConfig):
    system = KnownSystem.a4000x2
    multi_stream_expected_latency_ns = 1809322
~~~

The initial portion of `NVIDIA/code/common/systems/custom_list.json` showing `"host_memory"` as a field to ignore when matching:

~~~json
{
  "a4000x2": {
    "__obj_class__": "<class 'nvmitten.system.component.Description'>",
    "data": {
      "_component_module": "nvmitten.system.system",
      "_component_classname": "System",
      "_match_ignore_fields": ["host_memory"],
      // ...
    }
}
~~~

### Run the Benchmark

Run the offline scenario of the Resnet50 benchmark with Juice:

~~~sh
juice run make run RUN_ARGS="--benchmarks=resnet50 --scenarios=offline"
~~~

### Troubleshooting

**Why isn't my custom system recognized?**

The amount of available system RAM is used to determine part of a system's identity.  When I ran the benchmarks I ran into problems where the system I'd registered previously wasn't recognized because the amount of available system RAM had changed, e.g. some applications had started or stopped since I ran the Python script to add the custom system.

I fixed this by editing the host memory stored in `.../custom_systems.json` to match that reported in the no system found error when trying to run the benchmark.

**Why do I get the error "NameError: name 'Dict' is not defined"?**

The Python scripts generated to `.../configs/{benchmark}/{scenario}/custom.py` use the syntax `gpu_batch_size: Dict = {}` which isn't supported by the Python version in your Docker image.  Change to something like `gpu_batch_size = {'resnet50': 256}`.
