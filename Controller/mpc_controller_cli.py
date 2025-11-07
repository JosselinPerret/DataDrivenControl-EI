"""
Lightweight MPC Controller - Command Line Interface Version
For systems without GUI requirements or headless servers
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings
from scipy.optimize import minimize
import sys

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DT = 0.05  # Sampling time (seconds)
MPC_HORIZON = 10  # Prediction horizon (steps)
G = 9.81  # Gravity acceleration (m/s^2)
GLOBAL_MAX_ABS_Y = 19.62  # From training
X_MIN, X_MAX = 0.0, 1.0


# ============================================================================
# LSTM MODEL WRAPPER
# ============================================================================

class LSTMAccelerationModel:
    """Wrapper for LSTM model predictions"""
    
    def __init__(self, model_path):
        try:
            # Try loading normally first
            self.model = load_model(model_path)
        except ValueError as e:
            # If deserialization fails, load without compiling
            if "Could not deserialize" in str(e) or "not a KerasSaveable" in str(e):
                print("Note: Loading model with custom handling (compatibility mode)")
                self.model = load_model(model_path, compile=False)
            else:
                raise
        
        self.n_timesteps = self.model.input_shape[1]
        self.scaler = MinMaxScaler(feature_range=(X_MIN, X_MAX))
        dummy_data = np.array([[0.0], [1.0]])
        self.scaler.fit(dummy_data)
        
    def predict_acceleration(self, u_sequence):
        """Predict acceleration from input sequence"""
        u_scaled = self.scaler.transform(u_sequence.reshape(-1, 1)).flatten()
        u_reshaped = u_scaled.reshape(1, len(u_sequence), 1)
        y_normalized = self.model.predict(u_reshaped, verbose=0)
        a_pred = y_normalized[0, -1, 0] * GLOBAL_MAX_ABS_Y - G
        return a_pred


# ============================================================================
# MPC CONTROLLER
# ============================================================================

class LightweightMPCController:
    """Lightweight MPC controller for altitude control"""
    
    def __init__(self, lstm_model, horizon=MPC_HORIZON):
        self.lstm = lstm_model
        self.horizon = horizon
        self.dt = DT
        self.u_min = 0.0
        self.u_max = 1.0
        
    def predict_trajectory(self, u_sequence, x0):
        """Predict trajectory given input sequence"""
        trajectory = np.zeros((len(u_sequence) + 1, 2))
        trajectory[0] = x0
        
        for i in range(len(u_sequence)):
            u_window = np.ones(self.lstm.n_timesteps) * u_sequence[i]
            a = self.lstm.predict_acceleration(u_window)
            
            h_new = trajectory[i, 0] + trajectory[i, 1] * self.dt + 0.5 * a * self.dt**2
            v_new = trajectory[i, 1] + a * self.dt
            
            trajectory[i + 1, 0] = h_new
            trajectory[i + 1, 1] = v_new
        
        return trajectory
    
    def compute_control(self, x_current, h_ref, u_prev=None):
        """Compute optimal control input using MPC"""
        
        def objective(u_sequence):
            traj = self.predict_trajectory(u_sequence, x_current)
            h_error = (traj[1:, 0] - h_ref) ** 2
            cost = np.sum(h_error)
            
            if u_prev is not None:
                du = u_sequence[0] - u_prev
                cost += 0.1 * du**2
            
            cost += 0.01 * np.sum(u_sequence**2)
            cost += 0.05 * np.sum((u_sequence[1:] - u_sequence[:-1])**2)
            
            return cost
        
        u0 = u_prev if u_prev is not None else 0.5
        u_init = np.ones(self.horizon) * u0
        bounds = [(self.u_min, self.u_max) for _ in range(self.horizon)]
        
        result = minimize(
            objective,
            u_init,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-6}
        )
        
        return result.x[0]


# ============================================================================
# SIMULATION
# ============================================================================

def run_simulation(model_path, h_ref_trajectory, duration=10, verbose=True):
    """
    Run MPC simulation with given reference trajectory
    
    Args:
        model_path: Path to LSTM model
        h_ref_trajectory: Reference height trajectory (array or constant value)
        duration: Simulation duration (seconds)
        verbose: Print progress
    
    Returns:
        results: Dictionary with time history of all variables
    """
    
    # Initialize
    print("\n" + "="*70)
    print("LIGHTWEIGHT MPC CONTROLLER - SIMULATION")
    print("="*70)
    
    print(f"\nLoading model from: {model_path}")
    lstm_model = LSTMAccelerationModel(model_path)
    mpc = LightweightMPCController(lstm_model)
    
    print(f"Model loaded. Input timesteps: {lstm_model.n_timesteps}")
    print(f"MPC Horizon: {MPC_HORIZON} steps")
    print(f"Sampling time: {DT} s")
    print(f"Simulation duration: {duration} s\n")
    
    # State initialization
    x_current = np.array([0.0, 0.0])  # [height, velocity]
    u_current = 0.5  # Initial control input
    n_steps = int(duration / DT)
    
    # Convert h_ref to trajectory
    if isinstance(h_ref_trajectory, (int, float)):
        h_ref_traj = np.ones(n_steps) * h_ref_trajectory
    else:
        h_ref_traj = np.array(h_ref_trajectory)
        assert len(h_ref_traj) == n_steps, "Reference trajectory length mismatch"
    
    # Storage
    results = {
        'time': np.zeros(n_steps),
        'height': np.zeros(n_steps),
        'velocity': np.zeros(n_steps),
        'control': np.zeros(n_steps),
        'acceleration': np.zeros(n_steps),
        'reference': np.zeros(n_steps),
        'error': np.zeros(n_steps)
    }
    
    # Simulation loop
    print("Running simulation...")
    for i in range(n_steps):
        
        # Get reference for this step
        h_ref = h_ref_traj[i]
        
        # Compute MPC control
        u_opt = mpc.compute_control(x_current, h_ref, u_current)
        u_current = 0.7 * u_current + 0.3 * u_opt  # Smooth control
        
        # Predict acceleration
        u_window = np.ones(lstm_model.n_timesteps) * u_current
        a = lstm_model.predict_acceleration(u_window)
        
        # Update state
        h_new = x_current[0] + x_current[1] * DT + 0.5 * a * DT**2
        v_new = x_current[1] + a * DT
        x_current[0] = h_new
        x_current[1] = v_new
        
        # Store results
        results['time'][i] = i * DT
        results['height'][i] = h_new
        results['velocity'][i] = v_new
        results['control'][i] = u_current
        results['acceleration'][i] = a
        results['reference'][i] = h_ref
        results['error'][i] = h_new - h_ref
        
        # Print progress
        if verbose and (i % int(10/DT) == 0 or i == n_steps - 1):
            print(f"  Step {i}/{n_steps-1} | Time: {i*DT:6.2f}s | "
                  f"h: {h_new:6.2f}m | v: {v_new:6.2f}m/s | "
                  f"u: {u_current:5.2f} | error: {results['error'][i]:6.2f}m")
    
    print("\nSimulation complete!\n")
    
    return results


def plot_results(results):
    """Plot simulation results"""
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    time = results['time']
    
    # Height
    axes[0].plot(time, results['height'], 'b-', linewidth=2, label='Height')
    axes[0].plot(time, results['reference'], 'r--', linewidth=2, label='Reference')
    axes[0].fill_between(time, results['height'], results['reference'], 
                         alpha=0.2, color='gray')
    axes[0].set_ylabel('Height (m)', fontsize=11)
    axes[0].set_title('MPC Controller Performance', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best')
    
    # Velocity
    axes[1].plot(time, results['velocity'], 'g-', linewidth=2)
    axes[1].set_ylabel('Velocity (m/s)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Control Input
    axes[2].plot(time, results['control'], 'm-', linewidth=2)
    axes[2].set_ylabel('Control Input u', fontsize=11)
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3)
    
    # Acceleration & Error
    ax_a = axes[3]
    ax_a.plot(time, results['acceleration'], 'c-', linewidth=2, label='Acceleration')
    ax_a.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax_a.set_ylabel('Acceleration (m/s²)', fontsize=11)
    ax_a.set_xlabel('Time (s)', fontsize=11)
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(loc='upper left')
    
    # Add secondary axis for error
    ax_e = ax_a.twinx()
    ax_e.plot(time, results['error'], 'r:', linewidth=2, alpha=0.7, label='Height Error')
    ax_e.set_ylabel('Height Error (m)', fontsize=11, color='r')
    ax_e.tick_params(axis='y', labelcolor='r')
    
    fig.tight_layout()
    return fig


def print_statistics(results):
    """Print simulation statistics"""
    
    print("="*70)
    print("SIMULATION STATISTICS")
    print("="*70)
    
    h_error = results['error']
    
    print(f"\nHeight Performance:")
    print(f"  Final height: {results['height'][-1]:.3f} m")
    print(f"  Final reference: {results['reference'][-1]:.3f} m")
    print(f"  Mean error: {np.mean(np.abs(h_error)):.3f} m")
    print(f"  Max error: {np.max(np.abs(h_error)):.3f} m")
    print(f"  RMSE: {np.sqrt(np.mean(h_error**2)):.3f} m")
    
    print(f"\nControl Performance:")
    print(f"  Mean control input: {np.mean(results['control']):.3f}")
    print(f"  Max control input: {np.max(results['control']):.3f}")
    print(f"  Min control input: {np.min(results['control']):.3f}")
    print(f"  Control input std: {np.std(results['control']):.3f}")
    
    print(f"\nDynamics:")
    print(f"  Max velocity: {np.max(np.abs(results['velocity'])):.3f} m/s")
    print(f"  Max acceleration: {np.max(np.abs(results['acceleration'])):.3f} m/s²")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Model path
    model_path = './lstm_acceleration_model.h5'
    
    # Example 1: Step response (constant reference height)
    print("\n\n" + "#"*70)
    print("# EXAMPLE 1: STEP RESPONSE (h_ref = 5m)")
    print("#"*70)
    
    results_step = run_simulation(
        model_path=model_path,
        h_ref_trajectory=5.0,  # Constant 5m
        duration=15.0,
        verbose=True
    )
    
    print_statistics(results_step)
    
    # Example 2: Varying reference (dynamic trajectory)
    print("\n\n" + "#"*70)
    print("# EXAMPLE 2: DYNAMIC REFERENCE TRAJECTORY")
    print("#"*70)
    
    # Create a varying reference trajectory
    n_steps = int(20.0 / DT)
    time_ref = np.linspace(0, 20, n_steps)
    
    # Combine multiple height changes
    h_ref_dynamic = np.zeros(n_steps)
    h_ref_dynamic[0:int(100/DT)] = 5.0      # 0-5s: go to 5m
    h_ref_dynamic[int(100/DT):int(200/DT)] = 10.0  # 5-10s: go to 10m
    h_ref_dynamic[int(200/DT):int(300/DT)] = 3.0   # 10-15s: go to 3m
    h_ref_dynamic[int(300/DT):] = 7.0       # 15-20s: go to 7m
    
    results_dynamic = run_simulation(
        model_path=model_path,
        h_ref_trajectory=h_ref_dynamic,
        duration=20.0,
        verbose=True
    )
    
    print_statistics(results_dynamic)
    
    # Plot results
    print("Generating plots...")
    fig1 = plot_results(results_step)
    fig1.savefig('./mpc_step_response.png', dpi=150, bbox_inches='tight')
    print("  Saved: mpc_step_response.png")
    
    fig2 = plot_results(results_dynamic)
    fig2.savefig('./mpc_dynamic_response.png', dpi=150, bbox_inches='tight')
    print("  Saved: mpc_dynamic_response.png")
    
    plt.show()
