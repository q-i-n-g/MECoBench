# Modified VirtualHome Python API for MECoBench

This directory contains the modified VirtualHome Python API used by MECoBench.
It is installed with the same workflow as the official package:

```bash
pip install -e virtualhome
```

The package version remains `2.3.0` so existing VirtualHome import paths keep
working:

```python
from virtualhome.simulation.unity_simulator import UnityCommunication
```

Only the Unity communication API required by MECoBench is included here. Demo
notebooks, dataset generation utilities, evolving-graph simulation, and RL
environment code were removed from this release because they are not used by
the evaluation pipeline.

Use this Python API together with the MECoBench Simulator published on the
[GitHub releases page](https://github.com/q-i-n-g/MECoBench/releases). The
simulator release assets are named:

- `simulator_linux.zip`
- `simulator_MacOS_applesilicon.zip`
- `simulator_MacOS_intel.zip`
- `simulator_windows32.zip`
- `simulator_windows64.zip`
