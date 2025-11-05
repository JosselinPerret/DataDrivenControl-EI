# MPC (Model Predictive Control) Controller for Drone Altitude Control

This project provides two implementations of a lightweight Model Predictive Control (MPC) system for drone altitude control using a trained LSTM neural network model.

## Overview

The MPC controller predicts the acceleration output based on input commands using an LSTM model, then optimizes the control input to track a reference height in real-time. The system is lightweight and designed for resource-constrained environments.

**Key Features:**
- ✅ Lightweight MPC using SLSQP optimizer
- ✅ LSTM-based acceleration prediction
- ✅ Real-time GUI with reference height adjustment (GUI version)
- ✅ Real-time visualization of system state
- ✅ CLI version for headless/server deployment
- ✅ Smooth control input filtering
- ✅ Dynamic reference trajectory support
- ✅ Configurable prediction horizon and sampling time

## System Architecture

```
Reference Height (h_ref)
        ↓
    MPC Controller
        ↓ (Computes optimal u)
    u (Control Input)
        ↓
    LSTM Model
        ↓ (Predicts acceleration)
    a (Acceleration)
        ↓
    Kinematic Integrator
        ↓ (Integrates twice)
    h (Height), v (Velocity)
        ↓
    State Feedback
```

## File Structure

```
FINAL/
├── lstm_acceleration_model.h5          # Trained LSTM model (load this file)
├── mpc_controller.py                   # GUI version (real-time interactive)
├── mpc_controller_cli.py               # CLI version (batch/headless)
├── mpc_controller_README.md            # This file
├── mpc_step_response.png               # Example output (step response)
└── mpc_dynamic_response.png            # Example output (dynamic reference)
```

## Installation

### Prerequisites

```bash
pip install numpy tensorflow scikit-learn scipy matplotlib
```

### For GUI Version (Optional)
The GUI version uses tkinter which is usually included with Python. If not:
```bash
# On Ubuntu/Debian
sudo apt-get install python3-tk

# On macOS
brew install python-tk

# On Windows
# tkinter is included with most Python distributions
```

## Usage

### Option 1: GUI Version (Interactive Real-Time Control)

```bash
python mpc_controller.py
```

**Features:**
- Real-time reference height adjustment via slider (0-20m)
- Live plots of height, velocity, control input, and acceleration
- Current state display with error monitoring
- Start/Stop/Reset buttons
- Configuration display

**Controls:**
1. Adjust the **Reference Height** slider to change the target height in real-time
2. Click **Start** to begin the control loop
3. Observe the system response in the plots
4. Click **Stop** to pause
5. Click **Reset** to reset to initial state

**Screenshots:**
The GUI window shows:
- **Left Panel:** Control interface with reference height slider and state display
- **Right Panel:** Real-time plots of all system variables

### Option 2: CLI Version (Batch/Headless)

```bash
python mpc_controller_cli.py
```

**Features:**
- Runs pre-defined simulation examples
- Generates performance statistics
- Saves output plots as PNG files
- No GUI required

**Output Files:**
- `mpc_step_response.png` - Step response (h_ref = 5m constant)
- `mpc_dynamic_response.png` - Dynamic reference trajectory

**Example Output:**
```
======================================================================
LIGHTWEIGHT MPC CONTROLLER - SIMULATION
======================================================================

Loading model from: ./lstm_acceleration_model.h5
Model loaded. Input timesteps: 312
MPC Horizon: 10 steps
Sampling time: 0.05 s
Simulation duration: 15 s

Running simulation...
  Step 0/299 | Time:   0.00s | h:   0.00m | v:   0.00m/s | u:  0.50 | error:  -5.00m
  Step 50/299 | Time:   2.50s | h:   2.15m | v:   1.02m/s | u:  0.65 | error:  -2.85m
  Step 100/299 | Time:   5.00s | h:   4.82m | v:   0.31m/s | u:  0.55 | error:  -0.18m
  ...
```

## Configuration

Edit the configuration section in either script to customize:

```python
DT = 0.05              # Sampling time (seconds)
MPC_HORIZON = 10       # Prediction horizon (steps)
G = 9.81              # Gravity acceleration (m/s²)
GLOBAL_MAX_ABS_Y = 19.62  # Model scaling parameter (from training)
```

## System Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sampling Time (Δt) | 0.05 s | 20 Hz update rate |
| MPC Horizon | 10 steps | 0.5 second look-ahead |
| Command Range | [0, 1] | Normalized input |
| Reference Range | [0, 20] m | Height limits |
| Optimizer | SLSQP | Sequential Least Squares Programming |
| Max Iterations | 50 | Per optimization step |

## Control Objective

The MPC minimizes the following cost function:

$$J = \sum_{i=1}^{H} (h_i - h_{ref})^2 + \lambda_u \sum_{i=1}^{H} u_i^2 + \lambda_{\Delta u} ||u_1 - u_{prev}||^2 + \lambda_{\Delta \Delta u} \sum_{i=1}^{H-1} (\Delta u_i)^2$$

Where:
- **First term:** Height tracking error (main objective)
- **Second term:** Control effort penalty (λ_u = 0.01)
- **Third term:** Control smoothness relative to previous input (λ_u = 0.1)
- **Fourth term:** Control rate limiting (λ_ΔΔu = 0.05)

## LSTM Model Details

The trained LSTM model:
- **Architecture:** 1 LSTM layer (32 units) + Dense output layer
- **Input:** Normalized command sequence (312 timesteps × 1 feature)
- **Output:** Predicted acceleration (real m/s²)
- **Scaling:** 
  - Input: MinMaxScaler [0, 1]
  - Output: Denormalized using global_max_abs_y = 19.62

## Performance Metrics

The controller provides the following statistics:

```
HEIGHT PERFORMANCE:
- Mean absolute error: typically < 1m
- Max error: typically < 2m
- RMSE: < 1.5m

CONTROL PERFORMANCE:
- Smooth control input (0.7*u_prev + 0.3*u_new)
- Mean input typically 0.4-0.6
- Response time: ~2-3 seconds to reach reference

DYNAMICS:
- Max velocity: typically < 2 m/s
- Max acceleration: typically < 5 m/s²
```

## Advanced Usage

### Custom Reference Trajectory (CLI)

Modify the example in `mpc_controller_cli.py`:

```python
# Create custom reference trajectory
n_steps = int(20.0 / DT)  # 20 seconds
h_ref_custom = np.zeros(n_steps)

# Segment 1: Climb to 8m over 5 seconds
h_ref_custom[0:int(5/DT)] = np.linspace(0, 8, int(5/DT))

# Segment 2: Hold at 8m for 5 seconds
h_ref_custom[int(5/DT):int(10/DT)] = 8.0

# Segment 3: Descend to 2m over 5 seconds
h_ref_custom[int(10/DT):int(15/DT)] = np.linspace(8, 2, int(5/DT))

# Run simulation
results = run_simulation(
    model_path='./lstm_acceleration_model.h5',
    h_ref_trajectory=h_ref_custom,
    duration=20.0,
    verbose=True
)
```

### Programmatic Access (Python API)

```python
from mpc_controller_cli import LSTMAccelerationModel, LightweightMPCController

# Load model
lstm_model = LSTMAccelerationModel('./lstm_acceleration_model.h5')
mpc = LightweightMPCController(lstm_model, horizon=10)

# Current state
x_current = np.array([0.0, 0.0])  # height=0m, velocity=0m/s
h_ref = 5.0  # reference height

# Compute control
u_opt = mpc.compute_control(x_current, h_ref, u_prev=None)

print(f"Optimal control input: {u_opt:.3f}")
```

## Troubleshooting

### Model Loading Error
```
FileNotFoundError: lstm_acceleration_model.h5
```
**Solution:** Ensure the model file is in the same directory as the script or provide the correct path.

### CUDA/GPU Warning
```
Could not load dynamic library 'libcudart.so.11.0'
```
**Solution:** This is normal if CUDA is not installed. The model will use CPU (slower but functional).

### Slow Performance on GUI
**Solution:** Reduce `MPC_HORIZON` from 10 to 5 to speed up optimization.

### Model Predictions Unrealistic
**Causes:**
- Incorrect `GLOBAL_MAX_ABS_Y` value
- Input outside training range [0, 1]
- Model file corrupted

**Solution:** Verify the model was trained correctly and check scaling parameters.

## Theory

### Model Predictive Control (MPC)

MPC is an optimization-based control strategy that:
1. Predicts future system behavior over a horizon using a model
2. Solves an optimization problem to find the best control input
3. Applies only the first control input
4. Repeats at each time step

**Advantages:**
- ✅ Handles constraints naturally
- ✅ Predicts future behavior
- ✅ Multi-variable control
- ✅ Intuitive formulation

**Disadvantages:**
- ⚠️ Requires real-time optimization
- ⚠️ Model accuracy critical
- ⚠️ Computational cost for long horizons

### LSTM for Dynamics Modeling

LSTM (Long Short-Term Memory) networks are used to learn the nonlinear acceleration dynamics:
$$a = f(u_1, u_2, ..., u_T; \theta)$$

Where:
- $u_t$ are past control inputs
- $\theta$ are learned weights
- $f$ is the LSTM model

This replaces traditional physics-based models with learned dynamics.

## Performance Tips

1. **Reduce horizon if too slow:**
   ```python
   MPC_HORIZON = 5  # Instead of 10
   ```

2. **Adjust control smoothing:**
   ```python
   # In control loop
   u_current = 0.5 * u_current + 0.5 * u_new  # More aggressive
   ```

3. **Tune cost function weights:**
   - Increase height error weight for faster tracking
   - Decrease control weight for aggressive control
   - Increase smoothness weight for smooth control

4. **Monitor optimization:**
   - Check convergence by printing optimization result
   - Reduce `maxiter` if too slow

## References

- **MPC Overview:** Rawlings, J. B., et al. "Model Predictive Control: Theory, Computation, and Design"
- **LSTM Networks:** Hochreiter & Schmidhuber (1997). "Long Short-Term Memory"
- **Neural Network Dynamics:** Levine, S., et al. "End-to-End Training of Deep Visuomotor Policies"

## License

This project is part of the DataDrivenControl-EI repository.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify LSTM model is properly trained
3. Ensure all dependencies are installed
4. Check configuration parameters match training setup

---

**Last Updated:** 2025-01-01
**Version:** 1.0
