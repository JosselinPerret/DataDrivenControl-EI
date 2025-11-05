# Data-Driven Drone Altitude Control System - Complete Guide

## 📋 Overview

This Jupyter notebook implements an integrated pipeline for **data-driven drone altitude control** combining:

1. **Data Preparation & Analysis** - Load, normalize, and visualize control data
2. **Exploratory Visualization** - Statistical plots, FFT analysis, phase planes
3. **System Identification** - Extract discrete-time state-space model (A,B,C,D matrices)
4. **Mathematical Framework** - LQI control theory with Kalman observers
5. **Neural Network Models** - LSTM, GRU, or Transformer for sequence learning
6. **Model Predictive Control (MPC)** - LQI-based trajectory tracking
7. **Interactive Simulation** - Design altitude references and see system response
8. **Performance Analysis** - Compute tracking metrics and generate visualizations

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
pip install torch  # PyTorch for deep learning
```

### Running the Notebook
1. Open `DroneAltitudeControl.ipynb` in Jupyter
2. Run cells sequentially from top to bottom
3. The notebook is divided into 9 logical sections

---

## 📖 Section-by-Section Guide

### **Section 1: Setup and Data Loading**
- Imports all required libraries
- Specifies device (GPU/CPU for PyTorch)
- Loads CSV files: `bdd_in_mat_05.csv` (inputs), `bdd_out_mat_05.csv` (outputs)

**What happens:**
- u_raw: Command inputs (shape: N × 1)
- y_raw: System outputs - accelerations (shape: N × 1)

### **Section 2: Data Preparation and Exploration**
- Computes statistics (mean, std, min, max)
- Creates visualization with:
  - Time series plots of inputs and outputs
  - Phase plane (u vs y relationship)
  - Histogram distributions

**Key metrics computed:**
```
Input statistics:   Mean, Std, Range
Output statistics:  Mean, Std, Range
Data duration:      T = N * Ts (e.g., 1000 * 0.05 = 50 sec)
```

**Output:** `01_data_exploration.png`

---

### **Section 3: Mathematical Framework - System Identification**

#### Theory
The drone altitude dynamics are modeled as a **discrete-time LTI system**:

$$x_{k+1} = Ax_k + Bu_k$$
$$y_k = Cx_k + Du_k$$

Where:
- **State** $x_k = [z_k, v_k, a_k]^T$ (position, velocity, acceleration)
- **Input** $u_k \in [-1,1]$ (normalized thrust command)
- **Output** $y_k = a_k$ (measured acceleration)

#### LQI Controller (with augmented state)
$$\chi_k = [x_k; \eta_k] \quad \text{where} \quad \eta_k = \int (z_{ref} - z_k) dt$$

The control law minimizes:
$$J = \sum_k (\chi_k^T Q \chi_k + u_k^T R u_k)$$

Solving the **Discrete Algebraic Riccati Equation (DARE)**:
$$P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q$$

Gives the feedback gain:
$$K = (R + B^T P B)^{-1} B^T P A$$

And the control law:
$$u_k = -K\hat{\chi}_k$$

#### What the code does:
1. Integrates acceleration → velocity → position
2. Constructs regressor matrix from past states/inputs
3. Uses least-squares to estimate system parameters
4. Builds A, B, C, D matrices

**Output:** Identified state-space matrices

---

### **Section 4: Sequence Learning - Model Architecture Selection**

This section trains a neural network to learn the system's input-output mapping.

#### Three Model Options:

**1. LSTM (Long Short-Term Memory)**
```python
MODEL_TYPE = "LSTM"
```
- **Best for:** Long-term dependencies, temporal patterns
- **Pros:** Excellent gradient flow, handles vanishing gradients
- **Cons:** More parameters, slower training
- **Use when:** You have sequences with important long-range interactions

**2. GRU (Gated Recurrent Unit)**
```python
MODEL_TYPE = "GRU"
```
- **Best for:** Similar to LSTM but simpler
- **Pros:** Faster training, fewer parameters, competitive accuracy
- **Cons:** May struggle with very long sequences
- **Use when:** You want LSTM benefits with faster training

**3. Transformer**
```python
MODEL_TYPE = "Transformer"
```
- **Best for:** Parallel processing, modern architecture
- **Pros:** Can process sequences in parallel, excellent for attention patterns
- **Cons:** Requires more data, higher memory
- **Use when:** You have large datasets and computational resources

#### How it works:
1. **Sequence Creation:** Creates input sequences of length 20 with both u and y history
   - Input: `[u_{k-20:k}, y_{k-20:k}]` (20 timesteps)
   - Output: `y_{k+1}` (next acceleration)

2. **Train/Test Split:** 80% train, 20% test

3. **Training Loop:**
   - Uses MSE loss
   - Adam optimizer with learning rate scheduling
   - 50 epochs with validation monitoring

**Output:** `02_training_history.png` (training curves)

---

### **Section 5: Model Predictive Control (MPC)**

The MPC controller computes optimal control inputs over a **prediction horizon**.

#### Architecture:
```
State: χ = [z, v, a, η]ᵀ
       ├─ z: altitude (position)
       ├─ v: velocity
       ├─ a: acceleration
       └─ η: integral of tracking error

Control Law: u = -K χ̂
             where K solves discrete Riccati equation
             
Constraints: -1 ≤ u ≤ 1 (saturation)
```

#### Method:
1. **State Update:** z_k measured, v and a estimated from dynamics
2. **Integrator:** Accumulates tracking error for I (integral) action
3. **Control:** Computes u = -K χ with saturation
4. **Propagation:** Updates state using system model

#### Key tuning parameters:
```python
Q = np.diag([100, 10, 1, 500])   # Cost weights: [z, v, a, η]
R = 1.0                           # Control effort weight
```

Higher Q values → more aggressive tracking (but more oscillation)
Higher R values → smoother, less aggressive control

---

### **Section 6: Interactive Control - Design Your Trajectory**

Define custom altitude references! The code creates a multi-step reference:

```python
# Example trajectory (modify as needed)
0-5s:   z_ref = 0 m    (hover at ground)
5-10s:  z_ref = 5 m    (climb to 5m)
10-15s: z_ref = 8 m    (climb to 8m)
15-20s: z_ref = 3 m    (descend to 3m)
```

To use a different trajectory, modify `z_ref_trajectory`:
```python
# Smooth ramp
z_ref_trajectory = np.linspace(0, 10, num_steps)

# Sinusoidal reference
z_ref_trajectory = 5 * (1 + np.sin(2*np.pi*time_axis/10))

# Custom waypoints
waypoints = [(0, 0), (5, 5), (10, 8), (15, 3)]
```

---

### **Section 7: Performance Analysis and Metrics**

Computes standard control performance metrics:

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **IAE** | $\int_0^T \|e(t)\| dt$ | Total absolute error - lower is better |
| **ISE** | $\int_0^T e(t)^2 dt$ | Penalizes large errors - smoother response |
| **Max Error** | $\max_t \|e(t)\|$ | Worst-case tracking error |
| **Settling Time** | $t$ s.t. $\|e(t)\| < 0.05 \max$ | Time to stabilize (5% criterion) |
| **Control Effort** | $\sum_k \|u_k\|$ | Total energy used - important for battery life |

---

### **Section 8: Comparison and Advanced Features**

Uses the trained neural network to predict future system behavior given current state.

**How it works:**
1. Extracts last 20 samples of inputs and outputs
2. Feeds to neural network: $\hat{y}_{k+1} = NN([u_{k-19:k}, y_{k-19:k}])$
3. Inverse-scales prediction to get actual acceleration

This can be used for **hybrid control**: Combine MPC + NN predictions for better performance.

---

### **Section 9: Summary and Next Steps**

Generates comprehensive report with:
- Data statistics
- Model architecture and training results
- System identification matrices
- Control performance metrics
- Saved file locations

---

## 🎛️ Customization Guide

### Change Model Architecture
```python
# In Section 4, find this line:
MODEL_TYPE = "LSTM"  # Change to "GRU" or "Transformer"
```

### Adjust Controller Tuning
```python
# In Section 5, modify Q and R:
Q = np.diag([100, 10, 1, 500])  # Increase for more aggressive tracking
R = np.array([[1.0]])            # Increase for smoother response
```

### Define Custom Trajectory
```python
# In Section 6, modify the reference:
z_ref_trajectory = np.zeros(num_steps)
# Define your custom trajectory here
```

### Adjust Training Parameters
```python
# In Section 4:
epochs = 50              # More epochs = better fit (but slower)
batch_size = 32         # Smaller = noisier gradients, larger = needs more memory
lr = 1e-3               # Learning rate (higher = faster but unstable)
```

---

## 📊 Understanding the Plots

### Plot 1: Data Exploration (`01_data_exploration.png`)
- **Top:** Input commands over time (varies between -1 and 1)
- **Middle:** System output (accelerations)
- **Bottom:** Phase plane showing u vs y relationship

### Plot 2: Training History (`02_training_history.png`)
- **Blue line:** Training loss (should decrease smoothly)
- **Red line:** Test/validation loss (should decrease with training)
- **Y-axis:** Log scale (exponential improvements visible)

### Plot 3: MPC Results (`03_mpc_control_results.png`)
- **Top-left:** Position tracking (red dashed = reference, blue = actual)
- **Top-right:** Control commands issued by MPC
- **Bottom-left:** Tracking error over time
- **Bottom-right:** Phase plane of error dynamics

---

## 🔬 Mathematical Details

### Discrete-Time State-Space Model
Given measured acceleration $a_k = y_k$:

$$\begin{bmatrix} z_{k+1} \\ v_{k+1} \\ a_{k+1} \end{bmatrix} = \begin{bmatrix} 1 & T_s & 0 \\ 0 & 1 & T_s \\ 0 & 0 & \rho \end{bmatrix} \begin{bmatrix} z_k \\ v_k \\ a_k \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \\ \alpha \end{bmatrix} u_k$$

Where:
- $T_s = 0.05$ s (sampling time)
- $\rho$ = acceleration decay factor (~0.9-0.99)
- $\alpha$ = control input gain

### LQI Augmented System
Adding integrator state $\eta_k$:

$$\chi_{k+1} = A_{aug} \chi_k + B_{aug} u_k$$

where $A_{aug} = \begin{bmatrix} A & 0 \\ 1 & 1 \end{bmatrix}$ and tracking error is $\eta_{k+1} = \eta_k + (z_k - z_{ref})$

### Kalman Observer
Estimates full state from acceleration measurement only:

$$\hat{\chi}_k^+ = \hat{\chi}_k^- + L(y_k - \hat{y}_k)$$

L is computed from dual Riccati equation with process/measurement noise covariances.

---

## 💾 Output Files

| File | Description |
|------|-------------|
| `01_data_exploration.png` | Initial data analysis plots |
| `02_training_history.png` | Neural network training curves |
| `03_mpc_control_results.png` | MPC control performance |
| `{MODEL}_model.pth` | Trained PyTorch model (replace {MODEL} with LSTM/GRU/Transformer) |
| `system_matrices.npz` | Identified A, B, C, D matrices |
| `scalers.pkl` | Data normalization parameters for loading later |

---

## 🐛 Troubleshooting

**Problem:** Model training doesn't converge
- **Solution:** Reduce learning rate, increase epochs, check data normalization

**Problem:** MPC controller oscillates
- **Solution:** Increase R value (penalize control effort more), reduce Q

**Problem:** Out of memory errors
- **Solution:** Reduce batch_size, reduce sequence length (SEQ_LENGTH)

**Problem:** GPU not detected
- **Solution:** Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 📚 References

- **LQI Control:** Åström & Wittenmark, "Computer-Controlled Systems"
- **MPC:** Camacho & Bordons, "Model Predictive Control"
- **System ID:** Ljung, "System Identification: Theory for the User"
- **RNN:** Goodfellow et al., "Deep Learning" (Ch. 10)

---

## 🎓 Learning Path

**Beginner:** Run notebook as-is, understand outputs
**Intermediate:** Modify trajectories, adjust tuning, compare models
**Advanced:** Add constraints, implement adaptive control, add disturbances

---

**Questions?** Check the embedded comments in each cell or refer to the control theory references above!
