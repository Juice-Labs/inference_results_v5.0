# Juice MLPerf Inference v5.0 Benchmarks

This is a clone of the official MLCommons repository providing the most recent MLPerf benchmarks and results.  The Juice benchmarks in `closed/Juice` are based on the NVIDIA-optimized benchmarks in `closed/NVIDIA` with changes to run with Juice.  The changes amount to installing and using Juice to dynamically attach GPUs in the Docker container for running benchmarks.

To run a benchmark using Juice copy the Juice archive `juice-gpu-linux.tar.gz` for the Juice build you want to benchmark to `closed/Juice/docker/common` then follow the same instructions for running the NVIDIA benchmarks but running commands from the `closed/Juice` directory instead of `closed/NVIDIA`.  The steps in setting up and running benchmarks are the same up until the final two steps of adding a custom system and running the benchmark -- launch these steps with `juice run` to attach a Juice GPU dynamically.  For example the last two steps of running the Resnet50 benchmark and Offline scenario with Juice:

~~~sh
# ...
juice login
juice run python3 -m scripts.custom_systems.add_custom_system
juice run make run RUN_ARGS="--benchmarks resnet50 --scenarios offline"
~~~

See [Running Juice MLPerf Inference v5.0 Benchmarks](closed/Juice/README.md) for more details.
