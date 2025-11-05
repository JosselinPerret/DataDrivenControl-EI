# Drone Altitude Controller - MATLAB Version

Conversion MATLAB de `simple_controller.py` (Python avec Tkinter).

## Description

Ce projet implémente un contrôleur d'altitude pour drone utilisant:
- **LSTM neural network** pour prédiction de l'accélération
- **PID + Feedforward** contrôle pour suivi d'altitude
- **Interface MATLAB** avec plots en temps réel

## Fichiers

### Core Classes

1. **`DroneController.m`** - Classe principale
   - Charge le modèle LSTM
   - Normalise les entrées (MinMaxScaler)
   - Compute PID+FF control
   - Simule la dynamique

2. **`DroneControllerGUI.m`** - GUI interactive (optionnel)
   - Interface Tkinter-like en MATLAB
   - Slider pour référence d'altitude
   - Boutons Start/Stop/Reset
   - Plots en temps réel

### Scripts

3. **`example_usage.m`** - Exemples d'utilisation
   - Test 1: Hover test (vérifier accélérations)
   - Test 2: Suivi d'altitude (plusieurs hauteurs)
   - Test 3: Changement dynamique de consigne

## Requirements

- MATLAB R2020b ou newer
- **Deep Learning Toolbox** (pour charger modèle Keras)
- `lstm_acceleration_model.h5` (modèle Keras)
- `bdd_in_mat_05.csv` (données d'entraînement pour normalisation)

## Installation

### 1. Installer Deep Learning Toolbox

```matlab
% Dans MATLAB:
doc importKerasNetwork
```

### 2. Convertir modèle Keras (si nécessaire)

MATLAB peut charger directement `.h5` Keras:

```matlab
net = importKerasNetwork('lstm_acceleration_model.h5', 'OutputLayerType', 'regression');
```

### 3. Placer les fichiers

```
FINAL/
├── DroneController.m              # Classe principale
├── DroneControllerGUI.m          # GUI interactive
├── example_usage.m                # Examples
├── MATLAB_usage_README.md         # Ce fichier
├── lstm_acceleration_model.h5    # Modèle (du Python)
└── bdd_in_mat_05.csv            # Données d'entraînement
```

## Usage

### Option 1: Scripts simples

```matlab
%% Test basique
run('example_usage.m')
```

Cela va:
1. Tester le modèle LSTM
2. Simuler suivi d'altitude (5m, 10m, 15m)
3. Tester changement dynamique de consigne

### Option 2: GUI Interactive

```matlab
%% Lancer la GUI
gui = DroneControllerGUI('./lstm_acceleration_model.h5');
```

Features:
- Slider pour ajuster consigne en temps réel
- Start/Stop/Reset buttons
- Plots live des 4 variables
- Display état courant

### Option 3: Simulation personnalisée

```matlab
%% Créer controller
controller = DroneController('./lstm_acceleration_model.h5');

%% Configurer
controller.h_ref = 10.0;      % Consigne 10m
controller.kp = 0.40;         % Gains PID
controller.ki = 0.12;
controller.kd = 0.15;

%% Simuler 30 secondes
controller.simulate(30);

%% Afficher résultats
controller.print_stats();
```

## Architecture Contrôleur

### PID + Feedforward

```
u = u_hover + kp*error + ki*∫(error) - kd*velocity
```

Où:
- **u_hover = 0.70** : commande d'équilibre (hovering)
- **kp = 0.40** : gain proportionnel (erreur position)
- **ki = 0.12** : gain intégral (erreur steady-state)
- **kd = 0.15** : amortissement (vitesse)

### Normalisation entrée (MinMaxScaler)

Le modèle LSTM a été entraîné sur données normalisées:

```
u_normalized = (u - min) / (max - min) * 2 - 1
```

Cette normalisation est automatiquement appliquée dans `predict_acceleration()`.

### Dénormalisation sortie

```
a = y_normalized * GLOBAL_MAX_ABS_Y - G
a = y_normalized * 19.62 - 9.81 [m/s²]
```

## Paramètres

### Configuration de base

```matlab
controller.DT = 0.05;           % 20 Hz sampling
controller.G = 9.81;            % Gravity
controller.GLOBAL_MAX_ABS_Y = 19.62;  % Scaler parameter
```

### Gains PID+FF

```matlab
controller.kp = 0.40;           % Proportional
controller.ki = 0.12;           % Integral
controller.kd = 0.15;           % Derivative
controller.u_hover = 0.70;      % Feedforward (hovering equilibrium)
```

Ajuster ces gains pour:
- **Augmenter kp** → réponse plus rapide (mais moins stable)
- **Augmenter ki** → moins d'erreur d'équilibre (mais oscillations)
- **Augmenter kd** → moins d'oscillation (mais perte de rapidité)

## Outputs du modèle LSTM

Test des accélérations prédites:

```
u = -1.00  →  a = -29.29 m/s²  (maximum descente)
u = -0.50  →  a = -21.68 m/s²
u =  0.00  →  a = -9.89 m/s²   (free fall, gravité)
u =  0.50  →  a = +2.11 m/s²
u =  0.70  →  a = +5.76 m/s²   (≈ hover)
u =  1.00  →  a = +9.58 m/s²   (maximum montée)
```

## Comparaison Python vs MATLAB

| Aspect | Python | MATLAB |
|--------|--------|--------|
| GUI | Tkinter | uicontrol/uipanel |
| Threading | `threading` | `timer` |
| Plots | `matplotlib` | native `plot` |
| Model | Keras/TensorFlow | `importKerasNetwork` |
| Scaler | `sklearn.preprocessing` | Custom implementation |

## Limitations MATLAB

1. **Deep Learning Toolbox required** - Keras model loading
2. **No async threading** - Timer-based updates instead
3. **Model format** - H5 files need conversion utility (included in DLT)
4. **Performance** - MATLAB slower than Python for same operations

## Troubleshooting

### Erreur: "Undefined function 'importKerasNetwork'"
→ Installer Deep Learning Toolbox

### Erreur: "Could not load model"
→ Vérifier chemin fichier `.h5`
→ Vérifier Deep Learning Toolbox version

### Modèle predictions erronées
→ Vérifier `data_min` et `data_max` (normalisation)
→ Vérifier `GLOBAL_MAX_ABS_Y = 19.62`

### GUI ne se met pas à jour
→ Vérifier `timer` est bien lancé
→ Vérifier `running` flag

## Performance

Temps típiques par itération:

| Opération | Temps |
|-----------|-------|
| Load model | ~2-3s |
| Predict acceleration | ~50-100ms |
| Control step | ~1-5ms |
| GUI update | ~10-50ms |

**Note**: MATLAB peut être plus lent que Python pour ce type de workload.

## Prochaines étapes

Pour améliorer:
1. **Compiler MATLAB en exe** pour distribution
2. **Utiliser ONNX model** au lieu de Keras
3. **Implémenter en Simulink** pour simulation hardware-in-the-loop
4. **Ajouter feedback sensoriel réel** (capteurs altitude, vitesse)

## References

- `DroneController.m` - Main controller class
- `example_usage.m` - Run this first!
- `DroneControllerGUI.m` - Interactive mode

## Authors

Converted from Python to MATLAB
Original: `simple_controller.py`
Conversion: 2025

## License

Same as original project
