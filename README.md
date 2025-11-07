# Data-Driven Drone Altitude Control System

A comprehensive control system for drone altitude management combining data processing, machine learning, and advanced control theory. This project implements an integrated pipeline from raw data through LSTM-based modeling to Model Predictive Control (MPC) with interactive GUI interfaces.

## 📋 Project Overview

This repository contains a complete implementation of a data-driven approach to drone altitude control, featuring:

- **Data Processing Pipeline**: CSV data loading, normalization, and exploratory analysis
- **Machine Learning**: LSTM, GRU, and Transformer architectures for system dynamics prediction
- **Control Theory**: State-space modeling, Kalman filtering, and LQI control design
- **Model Predictive Control**: Optimization-based altitude trajectory planning
- **Interactive GUI**: Real-time visualization and control with Tkinter
- **Hybrid Architecture**: Optional State-Space Models (SSM) for error correction

## 🗂️ Project Structure

```text
DataDrivenControl-EI/
├── Controller/                      # Main control implementations
│   ├── mpc_controller.py           # MPC controller with GUI
│   ├── simple_controller.py         # Baseline controller
│   ├── control_law.py              # LQI control law with optional SSM
│   ├── mpc_controller_cli.py        # Command-line MPC interface
│   ├── DroneAltitudeControl.ipynb   # Main analysis notebook
│   ├── ssm_and_lq_control.ipynb     # Control theory deep dive
│   ├── test_model.py               # Model validation tests
│   ├── test_pid.py                 # PID controller tests
│   ├── advanced_testing.py         # Comprehensive testing suite
│   ├── quickstart.py               # Quick setup and testing script
│   ├── lstm_acceleration_model.h5  # Pre-trained LSTM model
│   ├── lstm_model.mat              # MATLAB format model
│   └── env/                        # Python virtual environment
│
├── Model training/                  # Model training pipeline
│   ├── train_model.ipynb           # LSTM/GRU/Transformer training
│   └── model_compute.ipynb         # Model performance evaluation
│
├── Alice/                           # Research/development folder
│   ├── work_Alice.ipynb            # Experimental work
│   ├── pyproject.toml              # Alice subproject configuration
│   ├── lstm_acceleration_model.h5  # Trained model
│   └── README.md                   # Alice-specific documentation
│
├── Conversion/                      # Model format conversion
│   ├── convert_h5_to_mat.py        # H5 → MATLAB conversion
│   └── convert_simple.py           # Simplified conversion utilities
│
├── data/                            # Training and test data
│   ├── bdd_in_mat_05.csv           # Input commands (normalized)
│   └── bdd_out_mat_05.csv          # Output acceleration (m/s²)
│
├── MatLab Files/                    # MATLAB compatibility files
└── README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites

### 1. Prerequisites

- **Python 3.10+** (tested with 3.10)
- **Virtual Environment** (recommended)

### 2. Installation

Clone the repository and install dependencies:

```bash
cd DataDrivenControl-EI
python -m venv venv
```

Activate the virtual environment:

- **Windows (PowerShell)**: `.\venv\Scripts\Activate.ps1`
- **Windows (CMD)**: `venv\Scripts\activate.bat`
- **macOS/Linux**: `source venv/bin/activate`

Install required packages:

```bash
pip install -r requirements.txt
```

Or use the provided pyproject.toml:

```bash
pip install numpy tensorflow keras scikit-learn scipy matplotlib h5py
```

### 3. Run the Quick Start

```bash
cd Controller
python quickstart.py
```

This will:

- ✓ Check all dependencies
- ✓ Validate the LSTM model
- ✓ Test both MPC and simple controllers
- ✓ Generate sample trajectories

### 4. Launch the GUI Controller

```bash
python mpc_controller.py
```

Features:

- **Real-time trajectory visualization**
- **Interactive altitude reference adjustment**
- **Live state space plot (position vs velocity)**
- **Performance metrics display**

## 📊 Core Components

### Controller Implementations

#### MPC Controller (`mpc_controller.py`)

- **Type**: Model Predictive Control with LSTM acceleration prediction
- **GUI**: Tkinter-based interactive interface
- **Features**:
  - Prediction horizon: 5 steps (configurable)
  - Real-time reference height adjustment
  - State visualization: trajectory and phase plane
  - Performance metrics: tracking error, computational time

**Configuration**:

```python
DT = 0.05              # Sampling time (50 ms)
MPC_HORIZON = 5        # Prediction steps
G = 9.81              # Gravity acceleration (m/s²)
GLOBAL_MAX_ABS_Y = 19.62  # Max training acceleration
```

#### LQI Control Law (`control_law.py`)
- **Type**: Linear Quadratic Integral with error feedback
- **State Observer**: Kalman filter for state estimation from acceleration measurements
- **Optional SSM**: State-Space Model for error correction (hybrid architecture)
- **Features**:
  - Discrete-time LQI synthesis
  - Full state feedback
  - Integrator for zero steady-state error

**Control Law**:
$$u_k = -K x_k - K_i \sum_{j=0}^{k} e_j$$

Where:
- $x_k = [z, \dot{z}, a_{int}, e_{int}]^T$ (position, velocity, integrated acceleration, integrated error)
- $K$ = optimal feedback gain
- $K_i$ = integral action gain

#### Simple Controller (`simple_controller.py`)
- **Type**: Baseline proportional-derivative controller
- **Purpose**: Validation and comparison baseline
- **GUI**: Similar Tkinter interface to MPC

### Machine Learning Models

#### LSTM Acceleration Prediction
The system uses a trained LSTM model to predict future accelerations based on:
- **Input**: Historical commands and system state
- **Output**: Predicted acceleration (normalized)
- **Training Data**: `data/bdd_in_mat_05.csv` and `data/bdd_out_mat_05.csv`

**Model File**: `lstm_acceleration_model.h5` (pre-trained, ready to use)

#### Supported Architectures
- **LSTM**: Long Short-Term Memory (primary)
- **GRU**: Gated Recurrent Unit (alternative)
- **Transformer**: Multi-head attention mechanism (experimental)

### Training Pipeline

#### `train_model.ipynb`
Complete training workflow:
1. Data loading and normalization
2. Sequence generation (sliding window)
3. Train/test split
4. Model architecture definition
5. Training with validation monitoring
6. Performance evaluation and visualization
7. Model export (H5 and MATLAB formats)

#### `model_compute.ipynb`
Model evaluation and analysis:
- One-step prediction error
- Multi-step rollout performance
- Comparison across architectures
- Uncertainty quantification

## 📈 System Dynamics

### Drone Altitude System Model

**Discrete-time state-space representation**:
$$z_{k+1} = z_k + \dot{z}_k \Delta t$$
$$\dot{z}_{k+1} = \dot{z}_k + a_k \Delta t$$
$$a_k = f(u_k, \text{history})$$

Where:
- $z_k$ = altitude (meters)
- $\dot{z}_k$ = vertical velocity (m/s)
- $a_k$ = vertical acceleration (m/s²)
- $u_k$ = command input (normalized)
- $\Delta t$ = 0.05 seconds (sampling period)
- $f(\cdot)$ = LSTM neural network predictor

### Data Characteristics

**Input Commands** (`bdd_in_mat_05.csv`):
- Range: [-1, 1] (normalized)
- Mean: ~0
- Statistics: Captured from real drone flights

**Output Acceleration** (`bdd_out_mat_05.csv`):
- Range: [-19.62, 19.62] m/s²
- Corresponding to: [-2g, +2g] variations
- Sampling rate: 20 Hz (Δt = 50 ms)

## 🎮 Usage Examples

### Example 1: Simple Controller with GUI

```bash
cd Controller
python simple_controller.py
```

1. Click "Start" to begin simulation
2. Adjust "Reference Height" slider to set target altitude
3. Observe real-time trajectory and tracking performance

### Example 2: MPC Controller

```bash
python mpc_controller.py
```

1. Configure MPC horizon if needed
2. Set altitude reference
3. Monitor prediction vs actual trajectory
4. Analyze state-space plot (position-velocity phase plane)

### Example 3: Command-Line Interface

```bash
python mpc_controller_cli.py
```

For non-GUI batch processing and testing.

### Example 4: Train Your Own Model

Open `Model training/train_model.ipynb` in Jupyter:

```bash
jupyter notebook "Model training/train_model.ipynb"
```

Follow cells to:
1. Load your own data
2. Configure model architecture
3. Train and validate
4. Export trained model

## 🔧 Configuration & Tuning

### MPC Parameters

Edit `Controller/mpc_controller.py`:

```python
DT = 0.05              # Sampling time (decrease for faster response)
MPC_HORIZON = 5        # Prediction steps (increase for smoother control)
```

### LQI Control Gains

Edit `Controller/control_law.py`:

```python
Q = np.diag([...])     # State cost weights
R = np.diag([...])     # Control cost weights
```

Adjust Q and R to balance tracking performance and control effort.

### Neural Network Architecture

Edit `Model training/train_model.ipynb`:

```python
model = keras.Sequential([
    layers.LSTM(64, activation='relu', input_shape=(seq_len, 1)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output: acceleration prediction
])
```

## 📋 Testing

### Run Test Suite

```bash
cd Controller
python advanced_testing.py
```

This executes:
- Model validation tests
- Controller performance benchmarks
- Trajectory tracking evaluation
- Computational efficiency analysis

### Individual Tests

```bash
python test_model.py      # LSTM model validation
python test_pid.py        # PID controller tests
```

## 🔄 Model Conversion

### Convert H5 to MATLAB Format

```bash
cd Conversion
python convert_h5_to_mat.py
```

Generates `lstm_model.mat` for MATLAB compatibility.

## 📚 Key References

### Control Theory
- **MPC**: Quadratic Programming for real-time optimization
- **LQI**: Discrete-time Linear Quadratic Integral control
- **Kalman Filter**: Optimal state estimation from noisy measurements

### Machine Learning
- **LSTM**: Hochreiter & Schmidhuber (1997)
- **Sequence Modeling**: Supervised learning with sliding window
- **Normalization**: MinMax scaling (training: [-1, 1])

### System Identification
- **State-Space Models**: Observer design for altitude estimation
- **Hybrid Approach**: NN prediction + SSM correction

## 🛠️ Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥2.2.6 | Numerical computation |
| tensorflow | ≥2.20.0 | Deep learning framework |
| keras | ≥3.12.0 | Neural network API |
| scikit-learn | ≥1.7.2 | Preprocessing & ML utilities |
| scipy | ≥1.15.3 | Scientific computing |
| matplotlib | ≥3.10.7 | Visualization |
| h5py | ≥3.15.1 | HDF5 file handling |
| torch | ≥2.9.0 | Optional: PyTorch support |
| cvxpy | ≥1.7.3 | Optional: Convex optimization |

Install all at once:

```bash
pip install numpy tensorflow keras scikit-learn scipy matplotlib h5py torch cvxpy
```

## 📝 Project Structure Details

### Notebooks

- **`Controller/DroneAltitudeControl.ipynb`**: Main integrated notebook with complete pipeline
- **`Controller/ssm_and_lq_control.ipynb`**: Deep dive into control theory and SSM design
- **`Model training/train_model.ipynb`**: LSTM/GRU training and validation
- **`Model training/model_compute.ipynb`**: Performance analysis and metrics
- **`Alice/work_Alice.ipynb`**: Experimental research work

### Python Scripts

- **`mpc_controller.py`**: GUI-based MPC implementation
- **`simple_controller.py`**: Baseline proportional controller
- **`control_law.py`**: LQI control with Kalman filter
- **`mpc_controller_cli.py`**: Command-line interface
- **`quickstart.py`**: Setup and validation script
- **`advanced_testing.py`**: Comprehensive test suite
- **`test_model.py`**: Model validation
- **`test_pid.py`**: PID controller tests

## 🔍 Troubleshooting

### Issue: "Module not found" errors

```bash
pip install -r requirements.txt
# or
pip install numpy tensorflow keras scikit-learn scipy matplotlib h5py
```

### Issue: CUDA/GPU not available

The system will automatically fall back to CPU. To force CPU:

```python
# In code
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Issue: Model file not found

Ensure `lstm_acceleration_model.h5` is in the `Controller/` directory:

```bash
ls Controller/lstm_acceleration_model.h5
```

### Issue: GUI not starting

Install tkinter (usually comes with Python):

```bash
# Windows
# Usually included - if not, reinstall Python with tcl/tk option

# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS
# Usually included - if not, reinstall Python or use conda
```

## 🤝 Contributing

To contribute:

1. Create a feature branch
2. Implement changes with tests
3. Update documentation
4. Submit pull request

## 📄 License

[Specify your license here]

## 👤 Author

**Josselin Perret**

For questions or issues, please open an issue on GitHub.

---

## 🎯 Key Features Summary

✅ **End-to-End System**: From raw data to real-time control  
✅ **Multiple Control Algorithms**: MPC, LQI, simple baseline  
✅ **Interactive GUI**: Real-time visualization and tuning  
✅ **Pre-trained Models**: Ready-to-use LSTM weights  
✅ **Comprehensive Testing**: Validation and performance analysis  
✅ **Research-Grade Code**: Well-documented and modular  
✅ **MATLAB Compatible**: Model export for MATLAB integration  
✅ **Flexible Architecture**: Easy to extend and customize  

---

**Last Updated**: November 2025  
**Status**: Active Development
