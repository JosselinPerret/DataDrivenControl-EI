#!/usr/bin/env python3
"""
Loi de Contrôle LQI pour Drone d'Altitude (avec SSM optionnel)
==============================================================

Contrôle d'altitude avec observateur Kalman et intégrateur d'erreur.
Mesure: accélération seulement
État: [position, vitesse, accélération_intégrée, erreur_intégrée]

SSM optionnel : pour correction d'erreur de modèle (architecture hybrid)
"""

import numpy as np
from scipy.linalg import solve_discrete_are
import os
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SSMPredictor:
    """SSM optionnel pour prédiction et correction d'erreur."""
    
    def __init__(self, A, B, C, Ts=0.05):
        """
        Args:
            A: Matrice d'état (n×n)
            B: Matrice d'entrée (n×m)
            C: Matrice de sortie (p×n)
            Ts: Temps d'échantillonnage
        """
        self.A = A
        self.B = B
        self.C = C
        self.Ts = Ts
        self.state_dim = A.shape[0]
        self.x = np.zeros(self.state_dim)
    
    def forward(self, u):
        """Étape forward du SSM.
        
        Args:
            u: Commande (scalaire ou array)
        
        Returns:
            y_pred: Prédiction de sortie
        """
        y_pred = self.C @ self.x
        self.x = self.A @ self.x + self.B.flatten() * float(u)
        return float(y_pred)
    
    def reset(self):
        """Réinitialise l'état caché."""
        self.x = np.zeros(self.state_dim)
    
    @staticmethod
    def load_from_npz(filename):
        """Charge SSM depuis un fichier .npz.
        
        Le fichier doit contenir: 'A', 'B', 'C', 'Ts'
        """
        data = np.load(filename)
        return SSMPredictor(
            A=data['A'],
            B=data['B'],
            C=data['C'],
            Ts=float(data['Ts'])
        )
    
    @staticmethod
    def load_from_pt(filename):
        """Charge SSM depuis un fichier PyTorch .pt.
        
        Le fichier doit contenir un dict avec: 'A', 'B', 'C', 'Ts'
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Use .npz instead.")
        
        checkpoint = torch.load(filename, map_location='cpu')
        return SSMPredictor(
            A=checkpoint['A'].numpy() if hasattr(checkpoint['A'], 'numpy') 
              else checkpoint['A'],
            B=checkpoint['B'].numpy() if hasattr(checkpoint['B'], 'numpy') 
              else checkpoint['B'],
            C=checkpoint['C'].numpy() if hasattr(checkpoint['C'], 'numpy') 
              else checkpoint['C'],
            Ts=float(checkpoint['Ts'])
        )


class DroneAltitudeController:
    """Contrôleur LQI pour l'altitude du drone avec SSM optionnel."""
    
    def __init__(self, Ts=0.05, z_ref=10.0, ssm_model=None, ssm_correction_gain=0.1,
                 Ad=None, Bd=None, C=None, D=None):
        """
        Initialisation du contrôleur.
        
        Args:
            Ts: Temps d'échantillonnage (s)
            z_ref: Altitude de référence (m)
            ssm_model: Modèle SSM optionnel (SSMPredictor ou chemin vers fichier)
            ssm_correction_gain: Gain pour correction SSM (0-1)
            Ad, Bd, C, D: Matrices du système identifié (si None, utilise valeurs par défaut)
        """
        self.Ts = Ts
        self.z_ref = z_ref
        
        # Charger SSM si fourni
        self.ssm = None
        self.ssm_gain = ssm_correction_gain
        if ssm_model is not None:
            if isinstance(ssm_model, str):
                # Charger depuis fichier
                if ssm_model.endswith('.npz'):
                    self.ssm = SSMPredictor.load_from_npz(ssm_model)
                elif ssm_model.endswith('.pt'):
                    self.ssm = SSMPredictor.load_from_pt(ssm_model)
                else:
                    raise ValueError("Format fichier non reconnu (.npz ou .pt)")
                print(f"✓ SSM chargé depuis {ssm_model}")
            else:
                self.ssm = ssm_model
                print("✓ SSM fourni directement")
        
        # ========================================================
        # SYSTÈME IDENTIFIÉ (2×2 physique)
        # ========================================================
        # État: [position z, vitesse v]
        # Mesure: accélération a
        
        # Utiliser matrices fournies ou valeurs par défaut
        if Ad is None:
            # VALEURS PAR DÉFAUT (à adapter pour VOTRE drone!)
            self.Ad = np.array([[1.0, Ts],
                                [0.0, 0.95]])      # 0.95 = coefficient d'amortissement
        else:
            self.Ad = Ad
        
        if Bd is None:
            self.Bd = np.array([[0.0],
                                [Ts]])             # Input (acceleration)
        else:
            self.Bd = Bd
        
        if C is None:
            self.C = np.array([[0.0, 1.0]])        # Output: velocity → acceleration
        else:
            self.C = C
        
        if D is None:
            self.D = np.array([[1.0]])             # Feedthrough: acceleration
        else:
            self.D = D
        
        self.b = 0.0                               # Bias
        
        # ========================================================
        # SYSTÈME AUGMENTÉ POUR LQI
        # ========================================================
        # État augmenté: [z, v, a_int, eta]
        # où:
        #   z = position
        #   v = vitesse
        #   a_int = accélération intégrée (cumsum de a)
        #   eta = intégrateur d'erreur (∫(z - z_ref) dt)
        
        self.A_aug = np.array([
            [1.0, Ts,  0.0,  0.0],       # z_{k+1} = z_k + v_k*Ts
            [0.0, 0.95, Ts,  0.0],       # v_{k+1} = 0.95*v_k + a_k*Ts
            [0.0, 0.0,  1.0,  0.0],      # a_k = mesure (pas d'équation)
            [1.0, 0.0,  0.0,  1.0]       # eta_{k+1} = eta_k + (z_k - z_ref)
        ])
        
        self.B_aug = np.array([[0.0],
                               [0.0],
                               [1.0],    # a_k vient de la mesure
                               [0.0]])
        
        # Matrice de mesure (observe accélération seulement)
        self.C_meas = np.array([[0.0, 0.0, 1.0, 0.0]])
        
        # ========================================================
        # DESIGN LQR/LQI
        # ========================================================
        
        # Coûts
        Q = np.diag([100.0,    # Penalize position error
                     1.0,      # Small velocity penalty
                     0.01,     # Small acceleration penalty
                     1000.0])  # Heavy integrator penalty
        
        R = np.array([[1.0]])   # Input (control effort)
        
        # Résoudre l'équation de Riccati
        P = solve_discrete_are(self.A_aug, self.B_aug, Q, R)
        
        # Gain LQI: K = (R + B'PB)^(-1) B'PA
        self.K_lqi = np.linalg.inv(R + self.B_aug.T @ P @ self.B_aug) @ \
                     self.B_aug.T @ P @ self.A_aug
        
        # ========================================================
        # OBSERVATEUR KALMAN
        # ========================================================
        
        # Bruits
        Qn = 1e-5 * np.eye(4)           # Process noise
        Rn = np.array([[1e-2]])         # Measurement noise
        
        # Gain Kalman
        Pn = solve_discrete_are(self.A_aug.T, self.C_meas.T, Qn, Rn)
        self.L_kalman = (Pn @ self.C_meas.T) @ \
                        np.linalg.inv(self.C_meas @ Pn @ self.C_meas.T + Rn)
        
        # ========================================================
        # ÉTAT INITIAL
        # ========================================================
        
        self.x_hat = np.zeros(4)  # [z_hat, v_hat, a_hat, eta_hat]
        self.u_prev = 0.0
        
    def step(self, a_measured, z_ref_new=None):
        """
        Exécute une itération du contrôleur.
        
        Args:
            a_measured: Accélération mesurée (m/s²)
            z_ref_new: Nouvelle référence d'altitude (optionnel)
        
        Returns:
            u: Commande de contrôle
            z_hat: Position estimée
            v_hat: Vitesse estimée
            correction: Correction SSM appliquée (si SSM utilisé)
        """
        
        if z_ref_new is not None:
            self.z_ref = z_ref_new
        
        # ====== CORRECTION KALMAN ======
        # Innovation
        a_pred = self.C_meas @ self.x_hat
        innovation = a_measured - a_pred
        
        # Mise à jour estimée
        self.x_hat = self.x_hat + self.L_kalman @ innovation.reshape(-1, 1)
        self.x_hat = self.x_hat.flatten()
        
        # ====== INTÉGRATEUR D'ERREUR ======
        # Correction de référence: intégrateur doit tracker z_ref
        self.x_hat[3] = self.x_hat[3] + (self.x_hat[0] - self.z_ref)
        
        # ====== LOI DE CONTRÔLE LQI ======
        # u = -K @ x_hat
        u = -self.K_lqi @ self.x_hat.reshape(-1, 1)
        u = float(np.clip(u, -1.0, 1.0))  # Saturation
        
        # ====== CORRECTION SSM (optionnel) ======
        correction = 0.0
        if self.ssm is not None:
            # Prédire accélération avec SSM
            a_pred_ssm = self.ssm.forward(u)
            
            # Erreur de prédiction: SSM détecte ce que LQI ne modélise pas
            error_ssm = a_measured - a_pred_ssm
            
            # Correction au gain u
            correction = self.ssm_gain * error_ssm
            u = u + correction
            u = float(np.clip(u, -1.0, 1.0))  # Re-saturation
        
        # ====== PRÉDICTION (state update) ======
        # x_{k+1} = A @ x_k + B @ a_measured
        self.x_hat = self.A_aug @ self.x_hat.reshape(-1, 1) + \
                     self.B_aug * a_measured
        self.x_hat = self.x_hat.flatten()
        
        self.u_prev = u
        
        return u, self.x_hat[0], self.x_hat[1], correction
    
    def get_state(self):
        """Retourne l'état estimé complet."""
        return {
            'position': self.x_hat[0],
            'velocity': self.x_hat[1],
            'acceleration': self.x_hat[2],
            'integrator': self.x_hat[3]
        }
    
    def reset(self):
        """Réinitialise l'état."""
        self.x_hat = np.zeros(4)
        self.u_prev = 0.0
        if self.ssm is not None:
            self.ssm.reset()


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    print("="*70)
    print("SIMULATION: Drone d'Altitude avec Contrôle LQI")
    print("="*70)
    
    # ========================================================
    # Option 1: LQI seul (rapide, production)
    # ========================================================
    print("\n[Mode 1] LQI seul (sans SSM)")
    controller_lqi = DroneAltitudeController(Ts=0.05, z_ref=10.0)
    
    # ========================================================
    # Option 2: LQI + SSM (hybrid, plus robuste)
    # ========================================================
    # À décommenter si vous avez un fichier SSM exporté
    print("[Mode 2] LQI + SSM (optionnel)")
    print("         Pour charger un SSM, passer: ssm_model='path/to/ssm.npz'")
    print("         ou créer un SSMPredictor manuellement")
    
    # Exemple : créer un SSM synthétique pour la démo
    if True:  # Mettre False si vous avez un vrai SSM à charger
        A_ssm = np.array([
            [0.9, 0.1],
            [0.0, 0.8]
        ])
        B_ssm = np.array([[0.5], [1.0]])
        C_ssm = np.array([[0.3, 0.7]])
        ssm_demo = SSMPredictor(A_ssm, B_ssm, C_ssm, Ts=0.05)
        controller_hybrid = DroneAltitudeController(
            Ts=0.05, z_ref=10.0, 
            ssm_model=ssm_demo,
            ssm_correction_gain=0.1
        )
    
    # ========================================================
    # Simulation
    # ========================================================
    num_steps = 500
    dt = 0.05
    
    # Stocker trajectoires
    positions_lqi = []
    velocities_lqi = []
    accelerations_lqi = []
    commands_lqi = []
    corrections_lqi = []
    
    positions_hybrid = []
    velocities_hybrid = []
    accelerations_hybrid = []
    commands_hybrid = []
    corrections_hybrid = []
    
    times = []
    
    # État réel du système (partagé)
    z_real = 0.0
    v_real = 0.0
    
    print(f"\nRéférence: {controller_lqi.z_ref} m")
    print(f"Temps simulation: {num_steps * dt:.1f} s\n")
    
    for k in range(num_steps):
        # ========== MESURE ==========
        a_real = commands_lqi[-1] if commands_lqi else 0.0
        
        # Intégration Euler
        v_real = v_real + a_real * dt
        z_real = z_real + v_real * dt
        a_measured = a_real + np.random.normal(0, 0.01)
        
        # ========== CONTRÔLE LQI seul ==========
        u_lqi, z_est_lqi, v_est_lqi, corr_lqi = controller_lqi.step(a_measured)
        
        # ========== CONTRÔLE LQI + SSM ==========
        u_hybrid, z_est_hybrid, v_est_hybrid, corr_hybrid = controller_hybrid.step(a_measured)
        
        # ========== ENREGISTREMENT ==========
        positions_lqi.append(z_real)
        velocities_lqi.append(v_real)
        accelerations_lqi.append(a_real)
        commands_lqi.append(u_lqi)
        corrections_lqi.append(corr_lqi)
        
        positions_hybrid.append(z_real)
        velocities_hybrid.append(v_real)
        accelerations_hybrid.append(a_real)
        commands_hybrid.append(u_hybrid)
        corrections_hybrid.append(corr_hybrid)
        
        times.append(k * dt)
        
        if (k + 1) % 100 == 0:
            print(f"Step {k+1:3d} | z={z_real:6.2f}m | "
                  f"u_lqi={u_lqi:6.3f} | u_hybrid={u_hybrid:6.3f}")
    
    # ========================================================
    # RÉSULTATS
    # ========================================================
    print("\n" + "="*70)
    print("RÉSULTATS")
    print("="*70)
    
    positions_lqi = np.array(positions_lqi)
    commands_lqi = np.array(commands_lqi)
    corrections_lqi = np.array(corrections_lqi)
    
    commands_hybrid = np.array(commands_hybrid)
    corrections_hybrid = np.array(corrections_hybrid)
    
    times = np.array(times)
    
    print(f"\nLQI seul:")
    print(f"  Position finale: {positions_lqi[-1]:.2f} m (cible: 10.0 m)")
    print(f"  Erreur: {np.abs(positions_lqi[-1] - 10.0):.3f} m")
    print(f"  Effort: {np.sum(np.abs(commands_lqi)):.1f}")
    
    print(f"\nLQI + SSM:")
    print(f"  Position finale: {positions_hybrid[-1]:.2f} m (cible: 10.0 m)")
    print(f"  Erreur: {np.abs(positions_hybrid[-1] - 10.0):.3f} m")
    print(f"  Effort: {np.sum(np.abs(commands_hybrid)):.1f}")
    print(f"  Correction SSM moyenne: {np.mean(np.abs(corrections_hybrid)):.4f}")
    
    # ========================================================
    # VISUALISATION
    # ========================================================
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    
    # Colonne 1: LQI seul
    axes[0, 0].plot(times, positions_lqi, 'b-', linewidth=2)
    axes[0, 0].axhline(10.0, color='k', linestyle='--', alpha=0.5)
    axes[0, 0].fill_between(times, 9.5, 10.5, alpha=0.1)
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].set_title('LQI Seul: Tracking')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].plot(times, commands_lqi, 'purple', linewidth=1.5)
    axes[1, 0].set_ylabel('Commande u')
    axes[1, 0].set_title('LQI Seul: Commande')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[2, 0].plot(times, corrections_lqi, 'orange', linewidth=1.5)
    axes[2, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[2, 0].set_ylabel('Correction')
    axes[2, 0].set_xlabel('Temps (s)')
    axes[2, 0].set_title('LQI Seul: Correction SSM (=0)')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Colonne 2: LQI + SSM
    axes[0, 1].plot(times, positions_hybrid, 'g-', linewidth=2)
    axes[0, 1].axhline(10.0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].fill_between(times, 9.5, 10.5, alpha=0.1)
    axes[0, 1].set_ylabel('Position (m)')
    axes[0, 1].set_title('LQI + SSM: Tracking')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].plot(times, commands_hybrid, 'red', linewidth=1.5, 
                    label='Total', alpha=0.7)
    axes[1, 1].plot(times, commands_lqi, 'purple', linewidth=1, 
                    linestyle='--', label='LQI seul', alpha=0.5)
    axes[1, 1].set_ylabel('Commande u')
    axes[1, 1].set_title('LQI + SSM: Commande')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 1].plot(times, corrections_hybrid, 'orange', linewidth=1.5)
    axes[2, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[2, 1].set_ylabel('Correction SSM')
    axes[2, 1].set_xlabel('Temps (s)')
    axes[2, 1].set_title('LQI + SSM: Correction (α·error)')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Comparaison: LQI vs LQI+SSM', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('control_law_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Figure sauvegardée: control_law_comparison.png")
    
    plt.show()
    
    # ========================================================
    # Exportation des matrices identifiées (pour utilisation future)
    # ========================================================
    print("\n" + "="*70)
    print("EXPORT DES MATRICES")
    print("="*70)
    print("\nPour exporter les matrices depuis le notebook et les charger ici:")
    print("\n1. Dans le notebook, ajouter à la fin:")
    print("""
    # Export SSM trained model
    ssm_export = {
        'A': ssm_model.A.detach().cpu().numpy(),
        'B': ssm_model.B.detach().cpu().numpy(),
        'C': ssm_model.C.detach().cpu().numpy(),
        'Ts': 0.05
    }
    np.savez('ssm_matrices.npz', **ssm_export)
    """)
    print("\n2. Puis utiliser dans ce script:")
    print("   controller = DroneAltitudeController(ssm_model='ssm_matrices.npz')")
