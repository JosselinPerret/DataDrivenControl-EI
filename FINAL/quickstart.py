#!/usr/bin/env python3
"""
Quick Start Script for MPC Controller
Run this to test both implementations
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    print("\n" + "="*70)
    print("CHECKING DEPENDENCIES")
    print("="*70 + "\n")
    
    required = {
        'numpy': 'numpy',
        'tensorflow': 'tensorflow.keras',
        'sklearn': 'sklearn',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib'
    }
    
    missing = []
    for name, module in required.items():
        try:
            __import__(module)
            print(f"✓ {name:15} installed")
        except ImportError:
            print(f"✗ {name:15} MISSING")
            missing.append(name)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\nAll dependencies installed ✓\n")
    return True


def check_model_file():
    """Check if LSTM model file exists"""
    print("="*70)
    print("CHECKING MODEL FILE")
    print("="*70 + "\n")
    
    model_path = './lstm_acceleration_model.h5'
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024*1024)
        print(f"✓ Model file found: {model_path}")
        print(f"  Size: {size_mb:.1f} MB\n")
        return True
    else:
        print(f"✗ Model file NOT found: {model_path}")
        print("\nPlease ensure lstm_acceleration_model.h5 is in the current directory")
        print("You can obtain it from: model_compute.ipynb\n")
        return False


def run_cli_demo():
    """Run CLI demo"""
    print("="*70)
    print("RUNNING CLI DEMO")
    print("="*70 + "\n")
    
    try:
        exec(open('mpc_controller_cli.py').read())
    except Exception as e:
        print(f"Error running CLI demo: {e}\n")
        return False
    
    return True


def run_gui_demo():
    """Run GUI demo"""
    print("="*70)
    print("RUNNING GUI DEMO")
    print("="*70 + "\n")
    
    try:
        print("Launching GUI... Close the window to exit.\n")
        exec(open('mpc_controller.py').read())
    except Exception as e:
        print(f"Error running GUI: {e}\n")
        print("Note: GUI requires tkinter and display server")
        print("If running headless, use CLI version instead\n")
        return False
    
    return True


def main():
    """Main menu"""
    print("\n" + "#"*70)
    print("# MPC CONTROLLER QUICK START")
    print("#"*70 + "\n")
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies and try again.")
        return
    
    # Check model file
    if not check_model_file():
        print("Please ensure the model file is in the current directory.")
        return
    
    # Menu
    while True:
        print("\nOptions:")
        print("  1. Run CLI Demo (batch simulation)")
        print("  2. Run GUI Demo (interactive control)")
        print("  3. Show Help")
        print("  4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            print()
            run_cli_demo()
            print("\nCLI demo complete. Check for generated PNG files.")
            
        elif choice == '2':
            print()
            run_gui_demo()
            print("\nGUI demo closed.")
            
        elif choice == '3':
            print("""
HELP:

CLI Version (mpc_controller_cli.py):
  - Runs batch simulations with predefined trajectories
  - Generates PNG plots of results
  - No GUI required (good for headless servers)
  - Fast execution
  
GUI Version (mpc_controller.py):
  - Interactive real-time control
  - Adjust reference height in real-time with slider
  - Live visualization of all variables
  - Start/Stop/Reset buttons
  - Requires display server (X11, Wayland, Windows GUI, macOS)

Both versions:
  - Load the trained LSTM model
  - Use MPC to compute optimal control input
  - Integrate acceleration twice to get height trajectory
  - Support dynamic reference tracking

Model File:
  - lstm_acceleration_model.h5 (must be present)
  - Trained on drone altitude control data
  - Input: normalized command [0, 1]
  - Output: vertical acceleration [m/s²]

For detailed documentation, see: MPC_CONTROLLER_README.md
            """)
            
        elif choice == '4':
            print("\nGoodbye!\n")
            break
        
        else:
            print("Invalid option. Please select 1-4.")


if __name__ == "__main__":
    main()
