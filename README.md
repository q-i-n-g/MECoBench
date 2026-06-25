# MECoBench

Repository of MECoBench: A Systematic Study of Multimodal Agent Collaboration in Embodied Environments.

MECoBench evaluates multi-agent embodied collaboration in VirtualHome. This GitHub release keeps the evaluation code and small task examples:

- `data/examples/parallel_5.json`
- `data/examples/sequential_5.json`

The full `parallel.json` and `sequential.json` task sets should be downloaded
from [Hugging Face](https://huggingface.co/datasets/q-i-n-g/MECoBench) and
placed under `data/task/`.

Generated-data scripts, intermediate task-knowledge assets, local notebooks, private service URLs, and machine-specific launch scripts have been removed.

## Setup

Create the Python environment from the repository root:

```bash
conda env create -f environment.yml
conda activate mecobench
```

Or install the same Python dependencies into an existing environment:

```bash
pip install -r requirements.txt
```

`requirements.txt` installs the modified VirtualHome API in editable mode. You can also run the same step manually:

```bash
pip install -e virtualhome
```

After installation, `pip show virtualhome` should point to this repository's `virtualhome/` directory. Prepare the matching MECoBench Simulator from the GitHub release.

Create model/runtime configuration:

```bash
cp eval/agents/.env.example eval/agents/.env
```

Then edit `eval/agents/.env` with your OpenAI-compatible endpoint, model keys, and `MECOBENCH_SIMULATOR_BIN`.

## VirtualHome API

This repository includes the modified Python API based on [VirtualHome](https://github.com/xavierpuigf/virtualhome), as used by MECoBench. After installing it with `pip install -e virtualhome`, verify the import:

```bash
python -c "import virtualhome.simulation.unity_simulator as u; print(u.UnityCommunication)"
```

Use the modified package and released MECoBench Simulator together; the Python API includes MECoBench-specific changes such as multi-agent room restrictions and extra camera controls.

## MECoBench Simulator

Download the MECoBench Simulator from the [GitHub releases page](https://github.com/q-i-n-g/MECoBench/releases). The release is a modified VirtualHome v2.3.0 Unity executable used for MECoBench experiments.

Available release assets:

- Linux: `simulator_linux.zip`
- macOS Apple Silicon: `simulator_MacOS_applesilicon.zip`
- macOS Intel: `simulator_MacOS_intel.zip`
- Windows 32-bit: `simulator_windows32.zip`
- Windows 64-bit: `simulator_windows64.zip`

Linux is recommended for batch evaluation. Example:

```bash
mkdir -p simulator
wget https://github.com/q-i-n-g/MECoBench/releases/download/v1/simulator_linux.zip -O simulator/simulator_linux.zip
unzip simulator/simulator_linux.zip -d simulator/linux
find simulator/linux -type f -name '*.x86_64' -exec chmod +x {} \;
export MECOBENCH_SIMULATOR_BIN="$(find "$PWD/simulator/linux" -type f -name '*.x86_64' | head -n 1)"
```

`UNITY_BIN` is still accepted for backward compatibility, but `MECOBENCH_SIMULATOR_BIN` is the preferred name in this repository.

## Launch Simulator

Start the display server:

```bash
bash src/start_screen.sh
```

In another terminal, launch MECoBench Simulator instances:

```bash
export MECOBENCH_SIMULATOR_BIN=/path/to/linux.x86_64
export UNITY_BASE_PORT=8001
export UNITY_GPUS=0,1
export UNITY_NUM_PER_GPU=10
bash src/start_unity_multi_8gpu.sh
```

The evaluation code uses `UNITY_BASE_PORT` and `UNITY_NUM_SIMULATORS` to allocate cases to simulator ports.

## Run Evaluation

Run a single example task set:

```bash
VWAH_TASK_FILE=data/examples/parallel_5.json \
UNITY_BASE_PORT=8001 \
UNITY_NUM_SIMULATORS=20 \
python eval/main.py
```

Run both example task sets with the provided launcher:

```bash
bash eval/exp_scripts/test_strong_leader.sh
```

To run the full task sets, download them from Hugging Face first:

```bash
mkdir -p data/task
huggingface-cli download q-i-n-g/MECoBench task/parallel.json --repo-type dataset --local-dir .
huggingface-cli download q-i-n-g/MECoBench task/sequential.json --repo-type dataset --local-dir .
mv task/parallel.json data/task/parallel.json
mv task/sequential.json data/task/sequential.json
rmdir task
```

Then run:

```bash
VWAH_USE_FULL_TASKS=1 bash eval/exp_scripts/test_strong_leader.sh
```

Useful environment variables:

- `VWAH_TASKS`: comma-separated task sets for the launcher, default `parallel,sequential`.
- `VWAH_USE_FULL_TASKS`: set to `1` to use `data/task/*.json`; default uses `data/examples/*_5.json`.
- `VWAH_TASK_ID_LIST`: comma-separated case IDs, default `0..95`.
- `VWAH_OUTPUT_ROOT`: output directory root, default `outputs/<task>`.
- `VWAH_COMMUNICATION_MODE`: `base`, `no_comm`, `discuss_then_act`, `leader_worker`, or `shared_memory`.
- `VWAH_NUM_AGENTS`: number of agents, default `2`.
- `VWAH_STEPS_THRESHOLD`: max steps per case.
- `VLM_MAIN_MODEL`, `VLM_LEADER_MODEL`, `VLM_WORKER_MODEL`, `VLM_RESOLVE_MODEL`, `VLM_EMBEDDING_MODEL`: model aliases defined in `eval/agents/models.json`.

Results are written as per-case `result.json` files plus a batch-level `summary.json`.
