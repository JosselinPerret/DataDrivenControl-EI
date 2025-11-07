"""
Lightweight Model Predictive Control (MPC) for Drone Altitude Control
Using trained LSTM model for acceleration prediction
Allows real-time reference height adjustment via GUI
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DT = 0.05  # Sampling time (seconds)
MPC_HORIZON = 5  # Prediction horizon (steps) - reduced for faster computation
G = 9.81  # Gravity acceleration (m/s^2)

# Model scaling parameters (from training)
# These should match the training notebook
GLOBAL_MAX_ABS_Y = 19.62  # This is the max absolute acceleration difference used in training
X_MIN = -1.0  # MinMaxScaler range for input (CORRECTION: was 0.0)
X_MAX = 1.0

# ============================================================================
# LSTM MODEL WRAPPER
# ============================================================================

class LSTMAccelerationModel:
    """Wrapper for LSTM model predictions"""
    
    def __init__(self, model_path):
        """
        Initialize LSTM model
        
        Args:
            model_path: Path to the trained LSTM model (.h5 file)
        """
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
        
        self.n_timesteps = self.model.input_shape[1]  # Number of timesteps
        self.scaler = MinMaxScaler(feature_range=(X_MIN, X_MAX))
        
        # Fit scaler on typical range [0, 1] for normalized commands
        dummy_data = np.array([[0.0], [1.0]])
        self.scaler.fit(dummy_data)
        
    def predict_acceleration(self, u_sequence):
        """
        Predict acceleration from input sequence
        
        Args:
            u_sequence: Input command sequence (n_timesteps,) normalized [0, 1]
            
        Returns:
            acceleration_real: Predicted acceleration in m/s^2
        """
        try:
            # Ensure proper data types (float32 for Keras)
            u_sequence = np.asarray(u_sequence, dtype=np.float32)
            
            # If sequence is too short, pad with the last value to match model input
            if len(u_sequence) < self.n_timesteps:
                padding = np.full(self.n_timesteps - len(u_sequence), u_sequence[-1], dtype=np.float32)
                u_sequence = np.concatenate([padding, u_sequence])
            
            # Scale input
            u_scaled = self.scaler.transform(u_sequence.reshape(-1, 1)).flatten()
            u_scaled = np.asarray(u_scaled, dtype=np.float32)
            
            # Reshape for LSTM (batch_size=1, timesteps, features=1)
            u_reshaped = u_scaled.reshape(1, len(u_sequence), 1).astype(np.float32)
            
            # Predict
            y_normalized = self.model.predict(u_reshaped, verbose=0)  # (1, n_timesteps, 1)
            
            # Denormalize: reverse the operation from training
            # y_normalized = (y - y[:, 0]) / global_max_abs_y
            # So: y = y_normalized * global_max_abs_y + y[:, 0]
            # Since we want final acceleration, take the last value
            a_pred = float(y_normalized[0, -1, 0]) * GLOBAL_MAX_ABS_Y - G
            
            return a_pred
        
        except Exception as e:
            print(f"Error in predict_acceleration: {e}")
            import traceback
            traceback.print_exc()
            # Return hover command acceleration (0 m/s^2)
            return 0.0


# ============================================================================
# MPC CONTROLLER
# ============================================================================

class LightweightMPCController:
    """Lightweight MPC controller for altitude control"""
    
    def __init__(self, lstm_model, horizon=MPC_HORIZON):
        """
        Initialize MPC controller
        
        Args:
            lstm_model: LSTMAccelerationModel instance
            horizon: Prediction horizon (steps)
        """
        self.lstm = lstm_model
        self.horizon = horizon
        self.dt = DT
        self.u_min = -1.0  # Minimum normalized command (CORRECTION: was 0.0)
        self.u_max = 1.0   # Maximum normalized command
        
    def predict_trajectory(self, u_sequence, x0):
        """
        Predict trajectory given input sequence
        
        Args:
            u_sequence: Input command sequence (horizon,)
            x0: Initial state [height, velocity]
            
        Returns:
            trajectory: (horizon+1, 2) array with [height, velocity]
        """
        trajectory = np.zeros((len(u_sequence) + 1, 2))
        trajectory[0] = x0
        
        for i in range(len(u_sequence)):
            # Get acceleration from LSTM
            # We need a window of past inputs for LSTM
            # For simplicity, use current command repeated to fill window
            u_window = np.ones(self.lstm.n_timesteps) * u_sequence[i]
            a = self.lstm.predict_acceleration(u_window)
            
            # Kinematic update
            h_new = trajectory[i, 0] + trajectory[i, 1] * self.dt + 0.5 * a * self.dt**2
            v_new = trajectory[i, 1] + a * self.dt
            
            # Safety constraints: prevent negative height
            if h_new < 0:
                h_new = 0.0
                v_new = 0.0  # Stop at ground
            
            trajectory[i + 1, 0] = h_new
            trajectory[i + 1, 1] = v_new
        
        return trajectory
    
    def compute_control(self, x_current, h_ref, u_prev=None):
        """
        Compute optimal control input using MPC
        
        Args:
            x_current: Current state [height, velocity]
            h_ref: Reference height
            u_prev: Previous control input (for smoothness)
            
        Returns:
            u_opt: Optimal control input (scalar, normalized [-1, 1])
        """
        import time
        t_start = time.time()
        
        def objective(u_sequence):
            """Objective function to minimize"""
            # Predict trajectory
            traj = self.predict_trajectory(u_sequence, x_current)
            
            # Height tracking error (main objective)
            h_error = (traj[1:, 0] - h_ref) ** 2
            cost = np.sum(h_error)
            
            # Penalty for negative heights (hard constraint via penalty)
            negative_h = np.maximum(0, -traj[1:, 0])
            cost += 1000.0 * np.sum(negative_h**2)
            
            # Regularization: penalize large control changes
            if u_prev is not None:
                du = u_sequence[0] - u_prev
                cost += 0.1 * du**2
            
            # Penalty for control input (encourage lower energy)
            cost += 0.01 * np.sum(u_sequence**2)
            
            # Penalty for aggressive acceleration
            cost += 0.05 * np.sum((u_sequence[1:] - u_sequence[:-1])**2)
            
            return cost
        
        # Initial guess: maintain previous control or hover command
        u0 = u_prev if u_prev is not None else 0.5
        u_init = np.ones(self.horizon) * u0
        
        # Bounds
        bounds = [(self.u_min, self.u_max) for _ in range(self.horizon)]
        
        # Optimize (lightweight: use SLSQP method)
        result = minimize(
            objective,
            u_init,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-6}
        )
        
        # Return first control input in sequence
        u_opt = result.x[0]
        
        # Timing info (printed occasionally)
        t_elapsed = (time.time() - t_start) * 1000
        if hasattr(self, '_mpc_call_count'):
            self._mpc_call_count += 1
        else:
            self._mpc_call_count = 1
        
        if self._mpc_call_count % 10 == 0:  # Print every 10 calls
            print(f"MPC optimization took {t_elapsed:.1f}ms (iterations: {result.nit})")
        
        return u_opt


# ============================================================================
# GUI APPLICATION
# ============================================================================

class MPCControllerGUI:
    """Real-time MPC Controller GUI with visualization"""
    
    def __init__(self, root, model_path):
        """Initialize GUI"""
        self.root = root
        self.root.title("Drone Altitude MPC Controller")
        self.root.geometry("1400x800")
        
        # Load model
        print("Loading LSTM model...")
        self.lstm_model = LSTMAccelerationModel(model_path)
        self.mpc = LightweightMPCController(self.lstm_model)
        print(f"Model loaded. Input timesteps: {self.lstm_model.n_timesteps}")
        
        # State variables
        self.x_current = np.array([0.0, 0.0])  # [height, velocity]
        self.h_ref = 5.0  # Reference height
        self.u_current = 0.0  # Current control input (will reach positive for thrust)
        self.running = False
        
        # History for plotting
        self.time_history = []
        self.height_history = []
        self.velocity_history = []
        self.reference_history = []
        self.control_history = []
        self.acceleration_history = []
        self.current_time = 0.0
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup user interface"""
        
        # ---- Control Panel (Left) ----
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)
        
        # Title
        title = ttk.Label(control_frame, text="MPC Controller", font=("Arial", 14, "bold"))
        title.pack(pady=10)
        
        # Reference Height
        ttk.Label(control_frame, text="Reference Height (m):", font=("Arial", 10)).pack(anchor=tk.W)
        self.h_ref_var = tk.DoubleVar(value=5.0)
        h_ref_scale = ttk.Scale(
            control_frame, from_=0, to=20, orient=tk.HORIZONTAL,
            variable=self.h_ref_var, command=self.update_reference
        )
        h_ref_scale.pack(fill=tk.X, pady=5)
        self.h_ref_label = ttk.Label(control_frame, text="5.0 m", font=("Arial", 10))
        self.h_ref_label.pack(anchor=tk.W)
        
        # Current State Display
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        ttk.Label(control_frame, text="Current State:", font=("Arial", 11, "bold")).pack(anchor=tk.W)
        self.height_label = ttk.Label(control_frame, text="Height: 0.00 m", font=("Arial", 9))
        self.height_label.pack(anchor=tk.W, pady=2)
        self.velocity_label = ttk.Label(control_frame, text="Velocity: 0.00 m/s", font=("Arial", 9))
        self.velocity_label.pack(anchor=tk.W, pady=2)
        self.control_label = ttk.Label(control_frame, text="Control: 0.50", font=("Arial", 9))
        self.control_label.pack(anchor=tk.W, pady=2)
        self.acceleration_label = ttk.Label(control_frame, text="Accel: 0.00 m/s²", font=("Arial", 9))
        self.acceleration_label.pack(anchor=tk.W, pady=2)
        
        # Error
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        self.error_label = ttk.Label(control_frame, text="Height Error: 0.00 m", font=("Arial", 9, "bold"))
        self.error_label.pack(anchor=tk.W)
        
        # Control Buttons
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_control)
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_control, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.reset_button = ttk.Button(button_frame, text="Reset", command=self.reset_state)
        self.reset_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Info Panel
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        info_text = """
MPC Configuration:
• Horizon: {} steps
• Dt: {} s
• Model: LSTM
• Optimizer: SLSQP

Command Range: [-1.0, 1.0]
Reference Range: [0, 20] m
        """.format(MPC_HORIZON, DT)
        
        info_label = ttk.Label(control_frame, text=info_text, font=("Arial", 8), justify=tk.LEFT)
        info_label.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # ---- Plotting Area (Right) ----
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create figure with subplots
        self.fig = Figure(figsize=(10, 8), dpi=100)
        
        self.ax_h = self.fig.add_subplot(4, 1, 1)
        self.ax_v = self.fig.add_subplot(4, 1, 2)
        self.ax_u = self.fig.add_subplot(4, 1, 3)
        self.ax_a = self.fig.add_subplot(4, 1, 4)
        
        self.ax_h.set_ylabel('Height (m)', fontsize=9)
        self.ax_v.set_ylabel('Velocity (m/s)', fontsize=9)
        self.ax_u.set_ylabel('Control Input', fontsize=9)
        self.ax_a.set_ylabel('Acceleration (m/s²)', fontsize=9)
        self.ax_a.set_xlabel('Time (s)', fontsize=9)
        
        for ax in [self.ax_h, self.ax_v, self.ax_u, self.ax_a]:
            ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def update_reference(self, value):
        """Update reference height from slider"""
        self.h_ref = float(value)
        self.h_ref_label.config(text=f"{self.h_ref:.1f} m")
        
    def start_control(self):
        """Start control loop"""
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Start control thread
            control_thread = threading.Thread(target=self.control_loop, daemon=True)
            control_thread.start()
            
            # Schedule display updates in the main thread
            self.schedule_display_update()
    
    def stop_control(self):
        """Stop control loop"""
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def reset_state(self):
        """Reset to initial state"""
        self.stop_control()
        self.x_current = np.array([0.0, 0.0])
        self.u_current = 0.5
        self.current_time = 0.0
        self.time_history.clear()
        self.height_history.clear()
        self.velocity_history.clear()
        self.reference_history.clear()
        self.control_history.clear()
        self.acceleration_history.clear()
        self.update_display()
    
    def control_loop(self):
        """Main control loop (runs in separate thread)"""
        import time
        
        loop_count = 0
        print("Control loop started...")
        
        while self.running:
            try:
                loop_count += 1
                t_start = time.time()
                
                # Compute MPC control
                u_new = self.mpc.compute_control(self.x_current, self.h_ref, self.u_current)
                
                # Ensure control is in valid range [-1, 1]
                u_new = np.clip(u_new, -1.0, 1.0)
                self.u_current = 0.7 * self.u_current + 0.3 * u_new  # Smooth control input
                
                # Predict acceleration using LSTM
                u_window = np.ones(self.lstm_model.n_timesteps, dtype=np.float32) * self.u_current
                a = self.lstm_model.predict_acceleration(u_window)
                
                # Update state with safety constraints
                h_new = self.x_current[0] + self.x_current[1] * DT + 0.5 * a * DT**2
                v_new = self.x_current[1] + a * DT
                
                # Prevent negative height
                if h_new < 0:
                    h_new = 0.0
                    v_new = max(0.0, v_new)  # Prevent downward velocity at ground
                
                self.x_current[0] = h_new
                self.x_current[1] = v_new
                
                # Store history
                self.time_history.append(self.current_time)
                self.height_history.append(h_new)
                self.velocity_history.append(v_new)
                self.reference_history.append(self.h_ref)
                self.control_history.append(self.u_current)
                self.acceleration_history.append(a)
                
                self.current_time += DT
                
                # Debug output on first few iterations
                if loop_count <= 5:
                    t_elapsed = time.time() - t_start
                    print(f"[{loop_count}] h={h_new:.3f}m, v={v_new:.3f}m/s, u={self.u_current:.3f}, a={a:.3f}m/s², t={t_elapsed*1000:.1f}ms")
                
                # Sleep to maintain 20 Hz control frequency (DT=0.05s)
                sleep_time = max(0.01, DT * 0.8 - (time.time() - t_start))
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Error in control loop: {e}")
                import traceback
                traceback.print_exc()
                self.running = False
    
    def schedule_display_update(self):
        """Schedule GUI update in the main thread (thread-safe)"""
        if self.running:
            self.update_display()
            # Schedule next update in 100ms (10 updates per second for display)
            self.root.after(100, self.schedule_display_update)
        else:
            # Final update when stopping
            self.update_display()
    
    def update_display(self):
        """Update GUI display"""
        try:
            # Update state labels
            height_error = self.x_current[0] - self.h_ref
            
            self.height_label.config(text=f"Height: {self.x_current[0]:.2f} m")
            self.velocity_label.config(text=f"Velocity: {self.x_current[1]:.2f} m/s")
            self.control_label.config(text=f"Control: {self.u_current:.2f}")
            
            if len(self.acceleration_history) > 0:
                a_current = self.acceleration_history[-1]
                self.acceleration_label.config(text=f"Accel: {a_current:.2f} m/s²")
            
            error_color = "red" if abs(height_error) > 1.0 else "black"
            self.error_label.config(
                text=f"Height Error: {height_error:.2f} m",
                foreground=error_color
            )
            
            # Update plots
            if len(self.time_history) > 0:
                self.ax_h.clear()
                self.ax_h.plot(self.time_history, self.height_history, 'b-', linewidth=2, label='Height')
                self.ax_h.plot(self.time_history, self.reference_history, 'r--', linewidth=2, label='Reference')
                self.ax_h.set_ylabel('Height (m)', fontsize=9)
                self.ax_h.grid(True, alpha=0.3)
                self.ax_h.legend(loc='upper left', fontsize=8)
                
                self.ax_v.clear()
                self.ax_v.plot(self.time_history, self.velocity_history, 'g-', linewidth=2)
                self.ax_v.set_ylabel('Velocity (m/s)', fontsize=9)
                self.ax_v.grid(True, alpha=0.3)
                
                self.ax_u.clear()
                self.ax_u.plot(self.time_history, self.control_history, 'm-', linewidth=2)
                self.ax_u.set_ylabel('Control Input', fontsize=9)
                self.ax_u.set_ylim([0, 1])
                self.ax_u.grid(True, alpha=0.3)
                
                self.ax_a.clear()
                self.ax_a.plot(self.time_history, self.acceleration_history, 'c-', linewidth=2)
                self.ax_a.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
                self.ax_a.set_ylabel('Acceleration (m/s²)', fontsize=9)
                self.ax_a.set_xlabel('Time (s)', fontsize=9)
                self.ax_a.grid(True, alpha=0.3)
                
                self.fig.tight_layout()
                self.canvas.draw()
        
        except Exception as e:
            print(f"Error updating display: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    
    # Path to the trained model
    model_path = './lstm_acceleration_model.h5'
    
    # Create GUI
    root = tk.Tk()
    app = MPCControllerGUI(root, model_path)
    
    # Start GUI event loop
    root.mainloop()


if __name__ == "__main__":
    main()
