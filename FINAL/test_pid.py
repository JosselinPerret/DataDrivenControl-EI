"""
Diagnostic du contrôleur PID - Simulation sans GUI
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Constants
DT = 0.05  # 20 Hz
G = 9.81
GLOBAL_MAX_ABS_Y = 19.62

# Load model
try:
    model = load_model('./lstm_acceleration_model.h5')
except ValueError:
    model = load_model('./lstm_acceleration_model.h5', compile=False)

n_timesteps = model.input_shape[1]

# Load and fit scaler
data_in = np.loadtxt('../bdd_in_mat_05.csv', delimiter=',')
scaler = MinMaxScaler()
scaler.fit(data_in.reshape(-1, 1))

def predict_acceleration(u_value):
    """Predict acceleration for constant command"""
    u_sequence = np.ones(n_timesteps, dtype=np.float32) * u_value
    u_normalized = scaler.transform(u_sequence.reshape(-1, 1)).flatten().astype(np.float32)
    u_reshaped = u_normalized.reshape(1, n_timesteps, 1).astype(np.float32)
    y_normalized = model.predict(u_reshaped, verbose=0)
    y_norm_last = float(y_normalized[0, -1, 0])
    a = y_norm_last * GLOBAL_MAX_ABS_Y - G
    return a

# PID Controller
class PIDController:
    def __init__(self, kp=0.35, ki=0.05, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_error = 0.0
        
    def compute(self, h, v, h_ref):
        error = h_ref - h
        self.integral_error += error * DT
        self.integral_error = np.clip(self.integral_error, -1.0, 1.0)
        
        u = self.kp * error + self.ki * self.integral_error - self.kd * v
        u = np.clip(u, -1.0, 1.0)
        
        return u
    
    def reset(self):
        self.integral_error = 0.0

# Simulation
print("="*70)
print("PID CONTROLLER TEST - Drone Altitude Control")
print("="*70)

h_ref = 10.0  # Target height
h = 0.0
v = 0.0
u = 0.0
controller = PIDController(kp=0.35, ki=0.05, kd=0.1)

print(f"\nTarget height: {h_ref} m")
print(f"PID gains: kp=0.35, ki=0.05, kd=0.1")
print("\nSimulation (50 seconds):")
print("-"*70)
print(f"{'Step':>4} {'Time':>6} {'h':>8} {'v':>8} {'u':>8} {'a':>8} {'h_ref':>8} {'error':>8}")
print("-"*70)

for step in range(int(50 / DT)):
    # Control
    u_cmd = controller.compute(h, v, h_ref)
    u = 0.8 * u + 0.2 * u_cmd  # Smoothing
    
    # Predict acceleration
    a = predict_acceleration(u)
    
    # Update state
    h_new = h + v * DT + 0.5 * a * DT**2
    v_new = v + a * DT
    
    # Safety
    if h_new < 0:
        h_new = 0.0
        v_new = max(0.0, v_new)
    
    h = h_new
    v = v_new
    error = h_ref - h
    
    # Print every 20 steps (1 second)
    if step % 20 == 0:
        print(f"{step:4d} {step*DT:6.2f} {h:8.3f} {v:8.3f} {u:8.3f} {a:8.2f} {h_ref:8.3f} {error:8.3f}")

print("-"*70)
print(f"\nFinal state:")
print(f"  Height:        {h:.3f} m (target: {h_ref} m)")
print(f"  Velocity:      {v:.3f} m/s")
print(f"  Control input: {u:.3f}")
print(f"  Error:         {h_ref - h:.3f} m")
print(f"  Integral acc:  {controller.integral_error:.3f}")

# Test what happens if we stay at target
print(f"\n" + "="*70)
print("What control is needed to stay at h={:.1f}m (zero acceleration)?".format(h_ref))
print("="*70)

# Find u that gives a=0
for u_test in np.linspace(-1.0, 1.0, 21):
    a_test = predict_acceleration(u_test)
    print(f"u={u_test:+.2f} -> a={a_test:+.2f} m/s²", end="")
    if abs(a_test) < 0.1:
        print(" ← EQUILIBRIUM")
    else:
        print()

print("\n" + "="*70)
print("ANALYSIS:")
print("="*70)
print("- If drone reaches target but can't hover, the PID needs tuning")
print("- Check if integral term is accumulating properly")
print("- Check if u reaches the equilibrium value automatically")
