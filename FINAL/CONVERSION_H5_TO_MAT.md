# Convertir Modèle Keras H5 en MATLAB

## 📋 Vue d'ensemble

Vous avez un modèle Keras/TensorFlow (`.h5`) et vous voulez l'utiliser en **MATLAB** ou **Simulink**.

Il existe **4 méthodes** selon vos besoins:

| Méthode | Format | Avantages | Inconvénients | Cas d'usage |
|---------|--------|-----------|---------------|------------|
| **1** | `.mat` (SciPy) | ✅ Direct MATLAB | ❌ Poids bruts | Production MATLAB |
| **2** | `.npz` (NumPy) | ✅ Compression | ❌ Python seulement | Archivage |
| **3** | `.onnx` | ✅ Cross-platform | ⚠️ Conversion complexe | Simulink ONNX Import |
| **4** | Dossier `.json` | ✅ Production-ready | ❌ Plus lourd | Déploiement |

---

## Méthode 1: `.mat` (RECOMMANDÉ)

### Étape 1: Exécuter le script Python

```bash
cd C:\Users\josse\OneDrive\Documents\GitHub\DataDrivenControl-EI\FINAL
python convert_h5_to_mat.py
```

Choisir **option 1** dans le menu interactif:

```
CHOISIR MÉTHODE DE CONVERSION
=====================================
1. SciPy .mat (Recommandé pour MATLAB) ✅
2. H5 → NPZ (Compression)
3. Keras → ONNX (Format universel)
4. Export détaillé (Production)
5. Toutes les méthodes
0. Quitter

Choisir (0-5): 1
```

### Étape 2: Fichier généré

```
✅ Modèle sauvegardé: lstm_model.mat
   Taille: 12.5 MB
```

### Étape 3: Charger en MATLAB

```matlab
% Charger le fichier .mat
data = load('lstm_model.mat');

% Accéder aux poids
weights_names = fieldnames(data);
for i = 1:length(weights_names)
    W = data.(weights_names{i});
    fprintf('%s: shape %s\n', weights_names{i}, mat2str(size(W)));
end
```

**Résultat:**

```
model_type: [1×1 char]
input_shape: [1 2000 1]
output_shape: [1 312 1]
lstm_layer_weight_0: [2000×400]
lstm_layer_weight_1: [400×400]
lstm_layer_weight_2: [400×1]
...
```

---

## Méthode 2: `.npz` (Compression)

### Conversion

```bash
python convert_h5_to_mat.py
# Choisir option 2
```

### Charger en Python après

```python
import numpy as np

data = np.load('lstm_model.npz')
# Accès: data['lstm_layer_weight_0'], etc.
```

---

## Méthode 3: `.onnx` (Simulink)

### Conversion

```bash
python convert_h5_to_mat.py
# Choisir option 3
```

Peut nécessiter:

```bash
pip install tf2onnx onnx onnxruntime
```

### Importer dans Simulink

1. **Ouvrir Simulink**
2. **Menu: Add-Ons → Deep Learning ONNX Importer**
3. **Sélectionner: `lstm_model.onnx`**
4. **Générer bloc Simulink automatiquement** ✓

---

## Méthode 4: Export détaillé (Production)

### Conversion

```bash
python convert_h5_to_mat.py
# Choisir option 5 (toutes les méthodes)
```

Génère dossier `lstm_export/` avec:

```
lstm_export/
├── architecture.json          # Structure du modèle
├── inference_config.json      # Configuration inférence
└── weights/                   # Poids individuels
    ├── lstm_layer_weight_0.npy
    ├── lstm_layer_weight_1.npy
    └── ...
```

### Charger architecture

```matlab
display_architecture('lstm_export/architecture.json');
```

Résultat:

```
📐 Architecture du modèle:
   Type: Functional
   Input shape: [1 2000 1]
   Output shape: [1 312 1]
   Nombre de layers: 5
   
   Détail des couches:
      1. lstm_layer (LSTM)
         Poids: 2, shapes: [2000 400] [400 400]
      2. dense_1 (Dense)
         Poids: 2, shapes: [400 312]
      ...
```

---

## Recommandations

### 🎯 Pour contrôle drone simple

**Utiliser Méthode 1 + importKerasNetwork**

```matlab
% Option A: Deep Learning Toolbox (recommandé)
net = importKerasNetwork('lstm_acceleration_model.h5', 'OutputLayerType', 'regression');
prediction = predict(net, u_normalized);

% Option B: Fichier .mat exporté (fallback)
data = load('lstm_model.mat');
% Implémenter forward MATLAB manuellement
```

### 🎯 Pour Simulink bloc

**Utiliser Méthode 3 (ONNX Import)**

```
Simulink → Add-Ons → Deep Learning ONNX Importer
→ Import lstm_model.onnx
→ Auto-generate Simulink block
```

### 🎯 Pour production / déploiement

**Utiliser Méthode 4 (Export détaillé)**

```matlab
% Charger configuration
config = jsondecode(fileread('lstm_export/inference_config.json'));

% Charger poids individuels (normalisation, etc.)
for layer = 1:config.num_layers
    W = load(['lstm_export/weights/layer_' num2str(layer) '.npy']);
end
```

---

## Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'scipy'"

```bash
pip install scipy
```

### ❌ "Cannot find lstm_model.mat"

Vérifier que le script Python s'est exécuté correctement:

```bash
python convert_h5_to_mat.py
# Vérifier sortie, pas d'erreur
# Vérifier fichier créé: ls -lh lstm_model.mat
```

### ❌ "Erreur chargement .mat en MATLAB"

```matlab
% Vérifier contenu
data = load('lstm_model.mat');
disp(data)

% Si problème, utiliser .npz à la place et converter
```

### ❌ "Deep Learning Toolbox not available"

```matlab
% Alternative: importer manuellement les poids
data = load('lstm_model.mat');

% Ou utiliser Méthode 3 (ONNX)
importKerasNetwork('lstm_model.onnx', ...);
```

---

## Scripts MATLAB fournis

### `load_lstm_model.m`

Charge fichier `.mat` et extrait poids:

```matlab
model = load_lstm_model('lstm_model.mat');

% Accéder aux poids
W_lstm = model.weights_dict.lstm_layer_weight_0;
b_lstm = model.weights_dict.lstm_layer_weight_2;
```

### `display_architecture(json_file)`

Affiche architecture du modèle:

```matlab
display_architecture('lstm_export/architecture.json');
```

---

## Résumé commandes

### Python (conversion)

```bash
# Option interactive
python convert_h5_to_mat.py

# Option automatique (CLI)
python convert_h5_to_mat.py --all
```

### MATLAB (chargement)

```matlab
% Charger poids
model = load_lstm_model('lstm_model.mat');

% Ou charger H5 directement
net = importKerasNetwork('lstm_acceleration_model.h5');

% Ou afficher architecture
display_architecture('lstm_export/architecture.json');
```

---

## Fichiers générés

Après conversion:

```
FINAL/
├── lstm_model.mat           (Poids SciPy)
├── lstm_model.npz           (Archive NumPy)
├── lstm_model.onnx          (Format universel)
└── lstm_export/             (Export détaillé)
    ├── architecture.json
    ├── inference_config.json
    └── weights/
```

---

## Pour Simulink

### Option A: Import ONNX (Recommandé)

```
1. Ouvrir Simulink
2. Simulink → Add-Ons → Deep Learning ONNX Importer
3. Charger lstm_model.onnx
4. Bloc Simulink généré automatiquement
5. Connecter input/output
```

### Option B: Deep Learning Toolbox direct

```matlab
% Dans Simulink model callback
net = importKerasNetwork('lstm_acceleration_model.h5');
% Ajouter bloc LSTM
```

### Option C: S-Function custom

```matlab
% Wrapper MATLAB custom
% Utiliser poids depuis .mat pour inférence manuelle
```

---

## Questions fréquentes

**Q: Quel format choisir?**  
A: Méthode 1 (`.mat`) pour MATLAB pur, Méthode 3 (`.onnx`) pour Simulink.

**Q: Peut-on faire forward pass complet en MATLAB?**  
A: Oui, mais complexe. Mieux d'utiliser `importKerasNetwork()` ou bloc Simulink.

**Q: La conversion preserve-t-elle exactitude?**  
A: Oui, poids sont exports en float32, inférence identique.

**Q: Quelle est la taille fichier?**  
A: ~12 MB (même que .h5 original).

**Q: Peut-on utiliser directement dans Simulink?**  
A: Oui, via ONNX Import ou Deep Learning Toolbox.

---

**✅ Prêt? Exécuter:**

```bash
python convert_h5_to_mat.py
```
