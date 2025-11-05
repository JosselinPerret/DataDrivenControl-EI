# MPC Controller - Quick Reference

## One-Minute Setup

```bash
# 1. Install dependencies
pip install numpy tensorflow scikit-learn scipy matplotlib

# 2. Navigate to FINAL folder
cd FINAL

# 3. Run the controller
# Option A: Interactive GUI (real-time)
python mpc_controller.py

# Option B: CLI simulation (batch)
python mpc_controller_cli.py

# Option C: Quick start wizard
python quickstart.py
```

## File Summary

| File | Purpose | Usage |
|------|---------|-------|
| `mpc_controller.py` | **Interactive GUI** - Real-time control with slider | `python mpc_controller.py` |
| `mpc_controller_cli.py` | **CLI simulation** - Batch processing & automation | `python mpc_controller_cli.py` |
| `advanced_testing.py` | **Analysis tools** - Parameter tuning & performance | `python advanced_testing.py` |
| `quickstart.py` | **Setup wizard** - First-time user guide | `python quickstart.py` |
| `lstm_acceleration_model.h5` | **Trained LSTM model** - Neural network weights | (Auto-loaded) |

## GUI Instructions (mpc_controller.py)

```
1. LAUNCH:
   python mpc_controller.py

2. LEFT PANEL - Controls:
   - Drag "Reference Height" slider (0-20m)
   - Click "Start" button
   - Watch system track your reference in real-time
   - Click "Stop" to pause
   - Click "Reset" to restart

3. RIGHT PANEL - Live Plots:
   [Height vs Reference]  ← Main tracking plot
   [Velocity]            ← System velocity
   [Control Input]       ← Command to system (0-1)
   [Acceleration]        ← System acceleration

4. INFO DISPLAY:
   - Current height (m)
   - Current velocity (m/s)
   - Control input (normalized)
   - Height error (m)
   - Color changes red if error > 1m
```

## CLI Instructions (mpc_controller_cli.py)

```
1. LAUNCH:
   python mpc_controller_cli.py

2. AUTOMATIC EXAMPLES:
   - Example 1: Step response (h_ref = 5m, 15s)
   - Example 2: Dynamic trajectory (varies height)

3. OUTPUT FILES:
   - mpc_step_response.png
   - mpc_dynamic_response.png

4. PLOTS SHOW:
   - Top: Height tracking
   - 2nd: Velocity
   - 3rd: Control input
   - Bottom: Acceleration + error
```

## Configuration

Edit these values in the scripts:

```python
# Sampling and prediction
DT = 0.05              # 20 Hz (0.05 seconds per step)
MPC_HORIZON = 10       # 10 steps = 0.5 second look-ahead

# Adjust for slower/faster response:
# Smaller horizon = faster but less optimal
# Larger horizon = smoother but slower
```

## Performance Targets

✓ **Real-time:** < 50 ms per control step
✓ **Accuracy:** Mean error < 1 m
✓ **Speed:** Reaches reference in 2-3 seconds
✓ **Smoothness:** Control input 0.3-0.7 range

## Common Tasks

### Task: Change reference height in real-time
```
→ Use GUI version (mpc_controller.py)
→ Move the slider
→ System responds in real-time
```

### Task: Test different trajectories
```
→ Edit mpc_controller_cli.py
→ Modify h_ref_dynamic array
→ Run: python mpc_controller_cli.py
```

### Task: Tune MPC parameters
```
→ Use advanced_testing.py
→ Tests horizon, cost weights, etc.
→ Generates performance analysis plots
```

### Task: Use in your code
```python
from mpc_controller_cli import LSTMAccelerationModel, LightweightMPCController
import numpy as np

# Initialize
lstm = LSTMAccelerationModel('./lstm_acceleration_model.h5')
mpc = LightweightMPCController(lstm, horizon=10)

# Current state
h, v = 0.0, 0.0  # Initial: height=0m, velocity=0m/s
x = np.array([h, v])

# Compute control input
u = mpc.compute_control(x, h_ref=5.0, u_prev=None)

# Apply to system
# ... send u to drone/simulator ...
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Model file not found | Ensure `lstm_acceleration_model.h5` exists in same folder |
| `ModuleNotFoundError: tensorflow` | Run: `pip install tensorflow` |
| GUI window doesn't appear | Try CLI version: `python mpc_controller_cli.py` |
| Slow response | Reduce `MPC_HORIZON` from 10 to 5 |
| Weird predictions | Check `GLOBAL_MAX_ABS_Y = 19.62` matches training |

## Key Concepts

**MPC = Model Predictive Control**
- Uses LSTM to predict future acceleration
- Optimizes control input over 10-step horizon
- Applies first computed control
- Repeats every 50ms

**Control Loop:**
```
1. Get current state (h, v)
2. Compute optimal control u
3. Apply u to system
4. System produces acceleration a
5. Integrate to get new state
6. Go to step 1
```

**Why MPC?**
- ✓ Optimal tracking
- ✓ Predicts future behavior
- ✓ Handles constraints naturally
- ✓ Real-time capable

## Resources

- **Full Guide:** See `MPC_CONTROLLER_README.md`
- **Theory:** See `MPC_CONTROLLER_README.md` References
- **Training Data:** See `model_compute.ipynb`
- **Python API:** All source code is well documented
- **Examples:** See `mpc_controller_cli.py` examples section

## Quick Test

```bash
# Test everything in 30 seconds:
python -c "
from mpc_controller_cli import LSTMAccelerationModel, LightweightMPCController
import numpy as np

print('Loading model...')
lstm = LSTMAccelerationModel('./lstm_acceleration_model.h5')
mpc = LightweightMPCController(lstm)

print('Computing control...')
x = np.array([0.0, 0.0])  # Start at h=0m, v=0m/s
u = mpc.compute_control(x, h_ref=5.0)

print(f'✓ SUCCESS!')
print(f'Current state: height=0.0m, velocity=0.0m/s')
print(f'Reference: 5.0m')
print(f'Optimal control: {u:.3f}')
"
```

## Support

For detailed documentation, see:
- `MPC_CONTROLLER_README.md` - Complete reference
- `mpc_implementation_summary.md` - Implementation details
- Source code comments - Inline documentation

---

**Ready to use! Start with:** `python mpc_controller.py`
