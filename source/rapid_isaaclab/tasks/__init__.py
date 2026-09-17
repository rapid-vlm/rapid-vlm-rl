# Copyright (c) 2026, The RAPID Authors.
# SPDX-License-Identifier: MIT

"""Custom IsaacLab manipulation task environments for RAPID.

Importing this package registers all environments with gymnasium:
- Isaac-Soccer-Franka-v0
- Isaac-Sweep-Into-Franka-v0
- Isaac-Push-Button-Franka-v0
- Isaac-Open-Drawer-Franka-v1
- Isaac-Open-Window-Franka-v0
"""

from . import soccer, sweep_into, button, cabinet, window_open
