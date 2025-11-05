"""
Advanced MPC Configuration and Testing Utilities
Use this for parameter tuning and advanced analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from mpc_controller_cli import (
    LSTMAccelerationModel, 
    LightweightMPCController,
    run_simulation,
    print_statistics
)
import sys

# ============================================================================
# PARAMETER TUNING
# ============================================================================

def test_horizon_effect():
    """Test effect of MPC horizon on performance"""
    
    print("\n" + "="*70)
    print("TESTING MPC HORIZON EFFECT")
    print("="*70 + "\n")
    
    horizons = [3, 5, 10, 15, 20]
    results_list = []
    
    for horizon in horizons:
        print(f"\nTesting horizon = {horizon} steps...")
        
        # Modify MPC horizon
        lstm_model = LSTMAccelerationModel('./lstm_acceleration_model.h5')
        mpc = LightweightMPCController(lstm_model, horizon=horizon)
        
        # Quick test
        from mpc_controller_cli import DT
        x_current = np.array([0.0, 0.0])
        h_ref = 5.0
        
        u_opt = mpc.compute_control(x_current, h_ref)
        print(f"  First control input: {u_opt:.4f}")
        
        results_list.append((horizon, u_opt))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    horizons_array = np.array([r[0] for r in results_list])
    control_inputs = np.array([r[1] for r in results_list])
    
    ax.plot(horizons_array, control_inputs, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('MPC Horizon (steps)', fontsize=12)
    ax.set_ylabel('Computed Control Input', fontsize=12)
    ax.set_title('Effect of MPC Horizon on Control Output', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(horizons)
    
    plt.tight_layout()
    plt.savefig('./horizon_effect.png', dpi=150)
    print("\nSaved: horizon_effect.png")


def test_reference_tracking():
    """Test tracking of different reference trajectories"""
    
    print("\n" + "="*70)
    print("TESTING REFERENCE TRACKING PERFORMANCE")
    print("="*70 + "\n")
    
    from mpc_controller_cli import DT
    
    # Define test trajectories
    trajectories = {
        'Step 5m': 5.0,
        'Step 10m': 10.0,
        'Step 2m': 2.0,
    }
    
    results_dict = {}
    
    for name, h_ref in trajectories.items():
        print(f"\nTesting: {name}")
        
        results = run_simulation(
            model_path='./lstm_acceleration_model.h5',
            h_ref_trajectory=h_ref,
            duration=10.0,
            verbose=False
        )
        
        rmse = np.sqrt(np.mean(results['error']**2))
        max_error = np.max(np.abs(results['error']))
        
        results_dict[name] = {
            'results': results,
            'rmse': rmse,
            'max_error': max_error
        }
        
        print(f"  RMSE: {rmse:.3f}m, Max Error: {max_error:.3f}m")
    
    # Plot comparison
    fig, axes = plt.subplots(len(trajectories), 1, figsize=(12, 4*len(trajectories)))
    if len(trajectories) == 1:
        axes = [axes]
    
    for idx, (name, data) in enumerate(results_dict.items()):
        res = data['results']
        ax = axes[idx]
        
        ax.plot(res['time'], res['height'], 'b-', linewidth=2, label='Actual')
        ax.plot(res['time'], res['reference'], 'r--', linewidth=2, label='Reference')
        ax.fill_between(res['time'], res['height'], res['reference'], 
                        alpha=0.2, color='gray')
        ax.set_ylabel('Height (m)', fontsize=11)
        ax.set_title(f'{name} - RMSE: {data["rmse"]:.3f}m', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('./reference_tracking.png', dpi=150)
    print("\nSaved: reference_tracking.png")


def test_control_smoothness():
    """Analyze control input smoothness"""
    
    print("\n" + "="*70)
    print("ANALYZING CONTROL INPUT SMOOTHNESS")
    print("="*70 + "\n")
    
    results = run_simulation(
        model_path='./lstm_acceleration_model.h5',
        h_ref_trajectory=5.0,
        duration=10.0,
        verbose=False
    )
    
    control = results['control']
    time = results['time']
    
    # Compute control derivatives
    du_dt = np.gradient(control, time)
    d2u_dt2 = np.gradient(du_dt, time)
    
    # Statistics
    print(f"\nControl Input Statistics:")
    print(f"  Mean: {np.mean(control):.4f}")
    print(f"  Std:  {np.std(control):.4f}")
    print(f"  Min:  {np.min(control):.4f}")
    print(f"  Max:  {np.max(control):.4f}")
    
    print(f"\nControl Rate Statistics:")
    print(f"  Mean |du/dt|: {np.mean(np.abs(du_dt)):.4f}")
    print(f"  Max |du/dt|:  {np.max(np.abs(du_dt)):.4f}")
    
    print(f"\nControl Acceleration Statistics:")
    print(f"  Mean |d²u/dt²|: {np.mean(np.abs(d2u_dt2)):.4f}")
    print(f"  Max |d²u/dt²|:  {np.max(np.abs(d2u_dt2)):.4f}")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    
    axes[0].plot(time, control, 'b-', linewidth=2)
    axes[0].set_ylabel('Control Input u', fontsize=11)
    axes[0].set_title('Control Input Analysis', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    axes[1].plot(time, du_dt, 'g-', linewidth=2)
    axes[1].set_ylabel('Control Rate du/dt', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    axes[2].plot(time, d2u_dt2, 'r-', linewidth=2)
    axes[2].set_ylabel('Control Accel d²u/dt²', fontsize=11)
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('./control_smoothness.png', dpi=150)
    print("\nSaved: control_smoothness.png")


def test_computational_cost():
    """Analyze computational cost of MPC"""
    
    print("\n" + "="*70)
    print("ANALYZING COMPUTATIONAL COST")
    print("="*70 + "\n")
    
    import time
    from mpc_controller_cli import DT
    
    horizons = [5, 10, 15, 20]
    computation_times = []
    
    for horizon in horizons:
        print(f"\nTesting horizon = {horizon}...")
        
        lstm_model = LSTMAccelerationModel('./lstm_acceleration_model.h5')
        mpc = LightweightMPCController(lstm_model, horizon=horizon)
        
        x_current = np.array([0.0, 0.0])
        h_ref = 5.0
        
        # Measure computation time
        times = []
        for _ in range(10):
            t_start = time.time()
            _ = mpc.compute_control(x_current, h_ref)
            t_end = time.time()
            times.append(t_end - t_start)
        
        mean_time = np.mean(times) * 1000  # Convert to ms
        computation_times.append(mean_time)
        
        print(f"  Mean computation time: {mean_time:.2f} ms")
        print(f"  Real-time factor: {mean_time/50:.2f}x (DT=50ms)")
        
        if mean_time < 50:
            print(f"  ✓ Runs in real-time (< 50ms)")
        else:
            print(f"  ✗ Too slow for real-time (> 50ms)")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(horizons, computation_times, 'ro-', linewidth=2, markersize=8)
    ax.axhline(y=50, color='g', linestyle='--', linewidth=2, label='Real-time limit (50ms)')
    ax.set_xlabel('MPC Horizon (steps)', fontsize=12)
    ax.set_ylabel('Computation Time (ms)', fontsize=12)
    ax.set_title('Computational Cost vs MPC Horizon', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xticks(horizons)
    
    plt.tight_layout()
    plt.savefig('./computational_cost.png', dpi=150)
    print("\nSaved: computational_cost.png")


def test_robustness():
    """Test robustness to different initial conditions"""
    
    print("\n" + "="*70)
    print("TESTING ROBUSTNESS TO INITIAL CONDITIONS")
    print("="*70 + "\n")
    
    from mpc_controller_cli import DT
    
    initial_velocities = [-2.0, -1.0, 0.0, 1.0, 2.0]
    results_dict = {}
    
    for v0 in initial_velocities:
        print(f"\nTesting initial velocity = {v0:.1f} m/s...")
        
        lstm_model = LSTMAccelerationModel('./lstm_acceleration_model.h5')
        mpc = LightweightMPCController(lstm_model)
        
        # Simulate with custom initial velocity
        x_current = np.array([0.0, v0])
        h_ref = 5.0
        
        time_hist = []
        height_hist = []
        u_current = 0.5
        
        for i in range(200):
            u_opt = mpc.compute_control(x_current, h_ref, u_current)
            u_current = 0.7 * u_current + 0.3 * u_opt
            
            u_window = np.ones(lstm_model.n_timesteps) * u_current
            a = lstm_model.predict_acceleration(u_window)
            
            x_current[0] += x_current[1] * DT + 0.5 * a * DT**2
            x_current[1] += a * DT
            
            time_hist.append(i * DT)
            height_hist.append(x_current[0])
        
        results_dict[f'v0={v0:.1f}'] = {
            'time': np.array(time_hist),
            'height': np.array(height_hist)
        }
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(initial_velocities)))
    
    for (name, data), color in zip(results_dict.items(), colors):
        ax.plot(data['time'], data['height'], linewidth=2.5, label=name, color=color)
    
    ax.axhline(y=5.0, color='k', linestyle='--', linewidth=2, alpha=0.5, label='Reference')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Height (m)', fontsize=12)
    ax.set_title('Robustness to Different Initial Velocities', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig('./robustness.png', dpi=150)
    print("\nSaved: robustness.png")


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    print("\n" + "#"*70)
    print("# MPC CONTROLLER - ADVANCED TESTING UTILITIES")
    print("#"*70 + "\n")
    
    tests = {
        '1': ('Horizon Effect', test_horizon_effect),
        '2': ('Reference Tracking', test_reference_tracking),
        '3': ('Control Smoothness', test_control_smoothness),
        '4': ('Computational Cost', test_computational_cost),
        '5': ('Robustness Analysis', test_robustness),
    }
    
    while True:
        print("\nAvailable Tests:")
        for key, (name, _) in tests.items():
            print(f"  {key}. {name}")
        print("  6. Run All Tests")
        print("  7. Exit")
        
        choice = input("\nSelect test (1-7): ").strip()
        
        if choice in tests:
            name, func = tests[choice]
            print(f"\n{'='*70}")
            print(f"Running: {name}")
            print('='*70)
            try:
                func()
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '6':
            for name, func in tests.values():
                try:
                    func()
                    print()
                except Exception as e:
                    print(f"Error in {name}: {e}\n")
        
        elif choice == '7':
            print("\nGoodbye!\n")
            break
        
        else:
            print("Invalid option")


if __name__ == "__main__":
    main()
