"""
lqi_lqg_recipe.py

Recette clé-en-main pour construire un contrôleur LQI (LQ avec intégrateur d'erreur)
et un observateur de type Kalman (discret) quand seule l'accélération est mesurée.

Remplacez Ad, Bd, C, D, b, Ts par vos matrices identifiées (ex: chargées depuis
"ssm_discrete.pt" ou sauvegarde numpy).

Usage rapide:
    python lqi_lqg_recipe.py

Le script contient:
- construction du modèle augmenté (x, v, z)
- ajout de l'intégrateur d'erreur (eta)
- calcul du gain LQI via solve_discrete_are
- calcul du gain de Kalman discret
- fonction lqi_step() effectuant correction (Kalman), mise à jour intégrateur,
  calcul de la commande u et prédiction xhat_{k+1|k}

Auteur: fournie pour ton notebook / repo
"""

import os
import numpy as np
from numpy.linalg import solve
from scipy.linalg import solve_discrete_are

# Optional: torch load if you have saved a PyTorch dict with Ad/Bd/C/D/b
try:
    import torch
except Exception:
    torch = None

# ---------------------------------------------------------------------------
# Utility: charger matrices (optionnel)
# ---------------------------------------------------------------------------

def load_from_pt(path):
    """Si `path` existe et contient un dict torch avec keys 'Ad','Bd','C','D','b',
    charge et renvoie en numpy. Retourne None si impossible.
    """
    if torch is None:
        return None
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location='cpu')
    Ad = data.get('Ad')
    Bd = data.get('Bd')
    C  = data.get('C')
    D  = data.get('D')
    b  = data.get('b')
    if Ad is None:
        return None
    # Convert to numpy if tensor
    def t2n(x):
        return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.array(x)
    return t2n(Ad), t2n(Bd), t2n(C), t2n(D), float(b)

# ---------------------------------------------------------------------------
# 1) Construction du modèle augmenté
# ---------------------------------------------------------------------------

def build_augmented(Ad, Bd, C, D, b, Ts):
    """Construit Aaug (n+2 x n+2), Baug (n+2 x 1), Cmeas, Dmeas
    États augmentés ξ = [x; v; z], où v et z sont intégrations de l'accélération a.
    """
    n = Ad.shape[0]
    assert Ad.shape == (n, n)
    assert Bd.shape[0] == n and Bd.shape[1] in (1,)
    # Build Aaug
    Aaug = np.block([
        [Ad,                 np.zeros((n,1)), np.zeros((n,1))],
        [Ts * C,             np.array([[1.0]]), np.array([[0.0]])],
        [np.zeros((1,n)),    np.array([[Ts]]),  np.array([[1.0]])]
    ])

    Baug = np.vstack([Bd.reshape(n,1), Ts * D.reshape(1,1), np.array([[0.0]])])

    # Mesure: a_k uniquement
    Cmeas = np.hstack([C.reshape(1, n), np.zeros((1,2))])
    Dmeas = D.reshape(1,1).copy()

    return Aaug, Baug, Cmeas, Dmeas

# ---------------------------------------------------------------------------
# 2) Ajout intégrateur d'erreur -> état étendu chi = [x; v; z; eta]
# ---------------------------------------------------------------------------

def build_lqi_system(Aaug, Baug):
    """Ajoute l'état eta et construit A_lqi (n+3 x n+3) et B_lqi (n+3 x 1)
    On suppose que la référence r entre en feedforward (non dans la dynamique)
    """
    n2 = Aaug.shape[0]  # = n + 2
    # A_lqi = [[Aaug, zeros], [hstack([0.., 0, 1]), 1]]
    A_lqi = np.block([
        [Aaug,                     np.zeros((n2,1))],
        [np.hstack([np.zeros((1, n2-2)), np.array([[0.0, 1.0]])]), np.array([[1.0]])]
    ])
    # Ensure last row picks z_k term: position of z is index n2-1
    # But above construction should already place correctly; still enforce explicitly:
    # last row before last column should be zeros except column (n2-1) = 1 (z_k)
    A_lqi[-1, n2-1] = 1.0

    B_lqi = np.vstack([Baug, np.array([[0.0]])])
    return A_lqi, B_lqi

# ---------------------------------------------------------------------------
# 3) Calcul du gain LQI via ARE discret
# ---------------------------------------------------------------------------

def compute_lqi_gain(A_lqi, B_lqi, Q=None, R=None):
    """Renvoie K de taille (1, n+3) pour u = -K chi_hat + u_ff"""
    n = A_lqi.shape[0]
    if Q is None:
        Q = np.eye(n)
    if R is None:
        R = np.array([[1e-2]])
    P = solve_discrete_are(A_lqi, B_lqi, Q, R)
    K = solve(B_lqi.T @ P @ B_lqi + R, B_lqi.T @ P @ A_lqi)
    return K, P

# ---------------------------------------------------------------------------
# 4) Observateur de Kalman discret (gain L) pour estimer [x; v; z]
# ---------------------------------------------------------------------------

def compute_kalman_gain(A_est, C_est, Qn, Rn):
    """Calcul du gain de Kalman discret L (n+2, 1) via ARE sur A'.
    A_est: (n+2, n+2), C_est: (1, n+2)
    Qn: process noise covariance (n+2, n+2)
    Rn: measurement noise covariance (1,1)
    """
    P_kf = solve_discrete_are(A_est.T, C_est.T, Qn, Rn)
    denom = (C_est @ P_kf @ C_est.T + Rn)
    L = (P_kf @ C_est.T) @ np.linalg.inv(denom)
    return L

# ---------------------------------------------------------------------------
# 5) Boucle de contrôle: lqi_step
# ---------------------------------------------------------------------------

def lqi_step(xhat_aug, eta, u_prev, r, a_meas, A_est, B_est, C_est, D_est, b, L, K):
    """
    Une étape de LQI + Kalman.

    xhat_aug : np.array (n+2,) estimate [xhat; vhat; zhat]
    eta      : float, intégrateur d'erreur
    u_prev   : float, commande appliquée précédemment (ou feedforward)
    r        : float, consigne d'altitude actuelle
    a_meas   : float, accélération mesurée (IMU)
    A_est..  : matrices du modèle (Aaug, Baug, Cmeas, Dmeas)
    b        : bias scalair (gravité) utilisé dans mesure prédite
    L        : gain Kalman (n+2, 1)
    K        : gain LQI (1, n+3)

    Retourne: u, xhat_aug_next_pred, eta_next, a_pred
    """
    # 1) Correction Kalman (mesure accel)
    # yhat = C_est @ xhat_aug + D_est * u_prev + b
    yhat = (C_est @ xhat_aug.reshape(-1,1) + D_est * np.array([[u_prev]]) + b).item()
    innov = a_meas - yhat
    xhat_aug = xhat_aug + (L.flatten() * innov)

    # 2) Mise à jour intégrateur d'erreur (zhat = position estimate)
    zhat = xhat_aug[-1]  # position is last of [x...; v; z]
    eta = eta + (zhat - r)

    # 3) LQI: calcul commande
    chi_hat = np.concatenate([xhat_aug.flatten(), np.array([eta])])  # (n+3,)
    u = float((-K @ chi_hat.reshape(-1,1)).item())

    # Option: clip commande selon bornes
    u = float(np.clip(u, -1.0, 1.0))

    # 4) Prédiction d'état estimé au pas suivant (prédiction open-loop)
    xhat_aug = (A_est @ xhat_aug.reshape(-1,1) + B_est * np.array([[u]])).flatten()

    # Prédiction acceleration (pour information)
    a_pred = (C_est @ xhat_aug.reshape(-1,1) + D_est * np.array([[u]]) + b).item()

    return u, xhat_aug, eta, a_pred

# ---------------------------------------------------------------------------
# Example d'utilisation / demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Remplacez ci-dessous par chargement réel si possible
    Ts = 0.05

    # Try to load from ssm_discrete.pt (torch) if present
    loaded = load_from_pt('ssm_discrete.pt')
    if loaded is not None:
        Ad, Bd, C, D, b = loaded
        Bd = Bd.reshape(-1,1)
        print('Loaded matrices from ssm_discrete.pt')
    else:
        # Exemple minimal (toy) si rien n'est fourni: double intégrator latente
        n = 2
        Ad = np.array([[1.0, Ts],[0.0, 1.0]])  # toy
        Bd = np.array([[0.0],[Ts]])
        C  = np.array([[1.0, 0.0]])  # measure some linear combination -> this is toy
        D  = np.array([[0.0]])
        b  = -9.81 * 0.0  # no bias in toy
        print('Using toy example matrices (replace with identified matrices)')

    # Build augmented model
    Aaug, Baug, Cmeas, Dmeas = build_augmented(Ad, Bd, C, D, b, Ts)
    A_lqi, B_lqi = build_lqi_system(Aaug, Baug)

    # Dimensions
    n2 = Aaug.shape[0]
    print(f'A_lqi shape: {A_lqi.shape}, B_lqi shape: {B_lqi.shape}')

    # LQR weights (à ajuster selon besoin)
    Q = np.eye(A_lqi.shape[0]) * 1.0
    # Relativement faibles poids sur latents x
    Q[:Ad.shape[0], :Ad.shape[0]] *= 1e-2
    Q[Ad.shape[0]+0, Ad.shape[0]+0] *= 1e-2  # v
    Q[Ad.shape[0]+1, Ad.shape[0]+1] *= 10.0  # z
    Q[-1, -1] *= 50.0  # eta
    R = np.array([[1e-2]])

    K, P = compute_lqi_gain(A_lqi, B_lqi, Q=Q, R=R)
    print('Computed LQI gain K shape:', K.shape)

    # Kalman: tunings (à ajuster selon bruit IMU)
    Qn = 1e-4 * np.eye(n2)
    Rn = np.array([[2e-2]])
    L = compute_kalman_gain(Aaug, Cmeas, Qn, Rn)
    print('Computed Kalman gain L shape:', L.shape)

    # Initial states
    xhat_aug = np.zeros(n2)
    eta = 0.0
    u_prev = 0.0

    # Example: simuler une consigne r[k] constante et fake mesures a_meas
    Tsim = 200
    r = 5.0  # consigne altitude

    # For demo, generate synthetic true system to produce a_meas (toy)
    # We'll simulate true latent x_true using Ad,Bd and compute a_true = C x_true + D u + b
    x_true = np.zeros(Ad.shape[0])
    v = 0.0
    z = 0.0
    u = 0.0

    traj = {
        'u': [], 'z': [], 'a_meas': [], 'u_cmd': []
    }

    for k in range(Tsim):
        # True acceleration measurement (toy open-loop): a = C x + D u + b
        a_true = (C @ x_true.reshape(-1,1) + D * np.array([[u]]) + b).item()
        # Add measurement noise
        a_meas = a_true + np.random.normal(scale=np.sqrt(Rn.item()))

        # Compute LQI step
        u_cmd, xhat_aug, eta, a_pred = lqi_step(xhat_aug, eta, u_prev, r, a_meas,
                                               Aaug, Baug, Cmeas, Dmeas, b, L, K)

        # Apply u_cmd to true system (toy)
        x_true = (Ad @ x_true.reshape(-1,1) + Bd * np.array([[u_cmd]])).flatten()
        # update integrators v and z for logging (if needed) using a_true
        v = v + Ts * a_true
        z = z + Ts * v

        traj['u'].append(u_cmd)
        traj['z'].append(z)
        traj['a_meas'].append(a_meas)
        traj['u_cmd'].append(u_cmd)

        u_prev = u_cmd
        u = u_cmd

    print('\nDemo finished. Example trajectories stored in `traj`.')
    print('First 10 commands:', traj['u_cmd'][:10])

# Fin du script
