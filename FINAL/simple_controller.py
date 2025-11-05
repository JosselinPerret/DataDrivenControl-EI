"""
Simple Altitude Controller (sans MPC) - juste pour tester que tout fonctionne
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

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DT = 0.05  # Sampling time (seconds)
G = 9.81  # Gravity acceleration (m/s^2)
GLOBAL_MAX_ABS_Y = 19.62
X_MIN = -1.0
X_MAX = 1.0

# ============================================================================
# LSTM MODEL WRAPPER
# ============================================================================

class LSTMAccelerationModel:
    """Wrapper for LSTM model predictions with correct normalization"""
    
    def __init__(self, model_path):
        """Load LSTM model and initialize scalers (exact same as training)"""
        try:
            self.model = load_model(model_path)
        except ValueError:
            print("Loading model with compile=False (compatibility mode)")
            self.model = load_model(model_path, compile=False)
        
        self.n_timesteps = self.model.input_shape[1]
        
        # CRITICAL: Load training data and fit scaler EXACTLY like training
        try:
            data_in = np.loadtxt('../bdd_in_mat_05.csv', delimiter=',')
            print(f"Training data loaded: {data_in.shape}")
            print(f"Training data range: [{data_in.min():.4f}, {data_in.max():.4f}]")
        except FileNotFoundError:
            print("ERROR: Could not find ../bdd_in_mat_05.csv")
            print("Using fallback range [-1, 1]")
            data_in = np.array([[-1.0], [1.0]])
        
        # Fit scaler on ACTUAL training data (like model_compute.ipynb)
        self.scaler = MinMaxScaler()
        self.scaler.fit(data_in.reshape(-1, 1))
        
        print(f"LSTM model loaded: input_shape={self.model.input_shape}")
        print(f"Scaler fitted: min={self.scaler.data_min_[0]:.4f}, max={self.scaler.data_max_[0]:.4f}")
        
    def predict_acceleration(self, u_sequence):
        """
        Predict acceleration using exact training normalization
        
        Normalization:
        1. u_sequence in [-1, 1] → normalize with MinMaxScaler
        2. model outputs y_normalized
        3. a = y_normalized * GLOBAL_MAX_ABS_Y - G
        
        Args:
            u_sequence: command sequence in [-1, 1] (n_timesteps,)
            
        Returns:
            a: acceleration in m/s²
        """
        try:
            u_sequence = np.asarray(u_sequence, dtype=np.float32)
            
            # Pad to match training length if needed
            if len(u_sequence) < self.n_timesteps:
                padding = np.full(self.n_timesteps - len(u_sequence), u_sequence[-1], dtype=np.float32)
                u_sequence = np.concatenate([u_sequence, padding])
            elif len(u_sequence) > self.n_timesteps:
                u_sequence = u_sequence[:self.n_timesteps]
            
            # STEP 1: Normalize using the training scaler
            u_normalized = self.scaler.transform(u_sequence.reshape(-1, 1)).flatten()
            u_normalized = np.asarray(u_normalized, dtype=np.float32)
            
            # STEP 2: Predict
            u_reshaped = u_normalized.reshape(1, self.n_timesteps, 1).astype(np.float32)
            y_normalized = self.model.predict(u_reshaped, verbose=0)
            
            # Get last timestep
            y_norm_last = float(y_normalized[0, -1, 0])
            
            # STEP 3: Denormalize (exact inverse of training)
            # a = y_normalized * GLOBAL_MAX_ABS_Y - G
            a = y_norm_last * GLOBAL_MAX_ABS_Y - G
            
            return a
        
        except Exception as e:
            print(f"Prediction error: {e}")
            return -G


# ============================================================================
# SIMPLE FEEDBACK CONTROLLER
# ============================================================================

class SimpleFeedbackController:
    """PID+Feedforward feedback controller for altitude tracking"""
    
    def __init__(self, lstm_model, kp=0.40, kd=0.15, ki=0.12):
        """
        Args:
            lstm_model: LSTM model
            kp: Proportional gain
            kd: Derivative gain (damping)
            ki: Integral gain (corrects steady-state error)
        """
        self.lstm = lstm_model
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.integral_error = 0.0  # Accumulator for integral term
        self.dt = 0.05  # Time step (20 Hz)
        
        # Equilibrium: Find u that gives a=0 (hovering)
        # From model test: u_hover ≈ 0.700 gives a ≈ 0 m/s²
        self.u_hover = 0.70  # Feed-forward value for hovering
        
    def compute_control(self, height, velocity, h_ref):
        """
        PID+Feedforward control:
        u = u_hover + kp * error + ki * integral(error) - kd * velocity
        
        The feed-forward term u_hover ensures we start from the right equilibrium.
        The integral term accumulates errors to ensure convergence.
        
        Args:
            height: Current height (m)
            velocity: Current velocity (m/s)
            h_ref: Reference height (m)
            
        Returns:
            u: Control input in [-1, 1]
        """
        error = h_ref - height
        
        # Accumulate error for integral term
        self.integral_error += error * self.dt
        # Anti-windup: limit integral accumulation
        self.integral_error = np.clip(self.integral_error, -2.0, 2.0)
        
        # PID+FF control
        u = self.u_hover              # Feed-forward: hovering equilibrium
        u += self.kp * error          # Proportional term
        u += self.ki * self.integral_error  # Integral term (corrects steady-state)
        u -= self.kd * velocity       # Derivative term (damping)
        
        # Clamp to [-1, 1]
        u = np.clip(u, -1.0, 1.0)
        
        return u
    
    def reset(self):
        """Reset integral error accumulator"""
        self.integral_error = 0.0


# ============================================================================
# GUI APPLICATION
# ============================================================================

class SimpleControllerGUI:
    """Real-time Controller GUI with visualization"""
    
    def __init__(self, root, model_path):
        """Initialize GUI"""
        self.root = root
        self.root.title("Drone Altitude Simple Controller")
        self.root.geometry("1400x800")
        
        # Load model
        print("Loading LSTM model...")
        self.lstm_model = LSTMAccelerationModel(model_path)
        self.controller = SimpleFeedbackController(self.lstm_model, kp=0.15)
        print(f"Model loaded. Input timesteps: {self.lstm_model.n_timesteps}")
        
        # State variables
        self.x_current = np.array([0.0, 0.0])  # [height, velocity]
        self.h_ref = 5.0
        self.u_current = 0.0
        self.running = False
        
        # History
        self.time_history = []
        self.height_history = []
        self.velocity_history = []
        self.reference_history = []
        self.control_history = []
        self.acceleration_history = []
        self.current_time = 0.0
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup GUI layout"""
        # ---- Control Panel (Left) ----
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        
        ttk.Label(control_frame, text="SIMPLE CONTROLLER", font=("Arial", 14, "bold")).pack(anchor=tk.W)
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
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
        self.control_label = ttk.Label(control_frame, text="Control: 0.00", font=("Arial", 9))
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
SIMPLE CONTROLLER:
• Proportional Feedback
• u = 0.15 * error
• Velocity damping

Command Range: [-1.0, 1.0]
Reference Range: [0, 20] m
        """
        
        info_label = ttk.Label(control_frame, text=info_text, font=("Arial", 8), justify=tk.LEFT)
        info_label.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # ---- Plotting Area (Right) ----
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
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
            
            # Schedule display updates in main thread
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
        self.u_current = 0.0
        self.current_time = 0.0
        self.controller.reset()  # Reset PID integral accumulator
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
                
                # Compute control
                u_new = self.controller.compute_control(self.x_current[0], self.x_current[1], self.h_ref)
                u_new = np.clip(u_new, -1.0, 1.0)
                self.u_current = 0.8 * self.u_current + 0.2 * u_new
                
                # Predict acceleration using LSTM
                u_window = np.ones(self.lstm_model.n_timesteps, dtype=np.float32) * self.u_current
                a = self.lstm_model.predict_acceleration(u_window)
                
                # Update state
                h_new = self.x_current[0] + self.x_current[1] * DT + 0.5 * a * DT**2
                v_new = self.x_current[1] + a * DT
                
                # Safety
                if h_new < 0:
                    h_new = 0.0
                    v_new = max(0.0, v_new)
                
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
                
                # Debug output
                if loop_count <= 10:
                    t_elapsed = (time.time() - t_start) * 1000
                    print(f"[{loop_count}] h={h_new:.3f}m, v={v_new:.3f}m/s, u={self.u_current:+.3f}, a={a:+.2f}m/s², t={t_elapsed:.1f}ms")
                
                # Sleep
                sleep_time = max(0.001, DT * 0.8 - (time.time() - t_start))
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
            self.root.after(100, self.schedule_display_update)
        else:
            self.update_display()
    
    def update_display(self):
        """Update GUI display"""
        try:
            # Update state labels
            height_error = self.x_current[0] - self.h_ref
            
            self.height_label.config(text=f"Height: {self.x_current[0]:.2f} m")
            self.velocity_label.config(text=f"Velocity: {self.x_current[1]:.2f} m/s")
            self.control_label.config(text=f"Control: {self.u_current:+.2f}")
            
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
                self.ax_u.set_ylim([-1.2, 1.2])
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
    model_path = './lstm_acceleration_model.h5'
    
    root = tk.Tk()
    app = SimpleControllerGUI(root, model_path)
    
    root.mainloop()


if __name__ == "__main__":
    main()
