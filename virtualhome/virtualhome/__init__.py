"""Modified VirtualHome Python API used by MECoBench."""

from __future__ import annotations

import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SIMULATION_PATH = os.path.join(_CURRENT_DIR, "simulation")
if _SIMULATION_PATH not in sys.path:
    sys.path.insert(0, _SIMULATION_PATH)

from unity_simulator.comm_unity import UnityCommunication

__all__ = ["UnityCommunication"]
