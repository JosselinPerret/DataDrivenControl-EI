"""Quick test to check LSTM model output"""

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Configuration
GLOBAL_MAX_ABS_Y = 19.62
G = 9.81
X_MIN = 0.0
X_MAX = 1.0

# Load model
print("Loading model...")
try:
    model = load_model('./lstm_acceleration_model.h5')
except ValueError as e:
    if "Could not deserialize" in str(e) or "not a KerasSaveable" in str(e):
        print("Loading with compatibility mode...")
        model = load_model('./lstm_acceleration_model.h5', compile=False)
    else:
        raise

print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

n_timesteps = model.input_shape[1]
print(f"Timesteps: {n_timesteps}")

# Setup scaler
scaler = MinMaxScaler(feature_range=(X_MIN, X_MAX))
dummy_data = np.array([[0.0], [1.0]])
scaler.fit(dummy_data)

# Test different control inputs
print("\n=== Testing model predictions ===")
for u_test in [0.3, 0.5, 0.7, 1.0]:
    # Create input sequence
    u_sequence = np.ones(n_timesteps, dtype=np.float32) * u_test
    u_scaled = scaler.transform(u_sequence.reshape(-1, 1)).flatten()
    u_reshaped = u_scaled.reshape(1, n_timesteps, 1).astype(np.float32)
    
    # Predict
    y_normalized = model.predict(u_reshaped, verbose=0)
    a_pred = float(y_normalized[0, -1, 0]) * GLOBAL_MAX_ABS_Y - G
    
    print(f"u={u_test:.1f} -> a={a_pred:.3f} m/s² (net: {a_pred - (-G):.3f})")

print("\n=== Analysis ===")
print(f"Gravity: {G} m/s²")
print(f"Hover needs: ~{G:.2f} m/s² upward acceleration")
print(f"Max acceleration range: ±{GLOBAL_MAX_ABS_Y:.2f} m/s²")
