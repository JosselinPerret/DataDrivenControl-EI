# 🚀 Convertir H5 en MAT - Démarrage Rapide

## ✅ C'est FAIT! 

Votre modèle Keras a été **converti avec succès** en format MATLAB!

```
✓ Fichier créé: lstm_model.mat
  Taille: 0.0 MB
```

---

## 📥 Comment Charger en MATLAB

### **Option 1: Charger les poids bruts (.mat)**

```matlab
% Ouvrir MATLAB
cd 'C:\Users\josse\OneDrive\Documents\GitHub\DataDrivenControl-EI\FINAL'

% Charger le fichier .mat
data = load('lstm_model.mat');

% Afficher contenu
weights_names = fieldnames(data);
for i = 1:length(weights_names)
    fprintf('%s\n', weights_names{i});
end
```

**Résultat:**
```
input_shape
output_shape
num_layers
lstm_1_w0       (poids LSTM)
lstm_1_w1       (récurrence LSTM)
lstm_1_w2       (biais LSTM)
dense_1_w0      (poids Dense)
dense_1_w1      (biais Dense)
```

### **Option 2: Utiliser Deep Learning Toolbox (MEILLEUR)**

```matlab
% Si vous avez Deep Learning Toolbox
net = importKerasNetwork('lstm_acceleration_model.h5', ...
    'OutputLayerType', 'regression');

% Faire une prédiction
u_input = randn(1, 2000, 1);  % [batch=1, timesteps=2000, features=1]
y_pred = predict(net, u_input);
```

---

## 🎯 Pour Votre Contrôleur Drone

### LSTM Accélération dans MATLAB

```matlab
%% Charger modèle
net = importKerasNetwork('lstm_acceleration_model.h5', ...
    'OutputLayerType', 'regression');

%% Dans boucle de contrôle
function a_pred = predict_acceleration(u_command, net)
    % Normaliser l'entrée
    u_min = -1.0;
    u_max = 1.0;
    u_normalized = 2 * (u_command - u_min) / (u_max - u_min) - 1;
    
    % Créer séquence (2000 timesteps constant)
    u_sequence = repmat(u_normalized, 1, 2000, 1);  % [1, 2000, 1]
    
    % Prédire
    y_normalized = predict(net, u_sequence);
    y_norm_last = y_normalized(1, 2000, 1);
    
    % Dénormaliser
    GLOBAL_MAX_ABS_Y = 19.62;
    G = 9.81;
    a_pred = y_norm_last * GLOBAL_MAX_ABS_Y - G;
end
```

---

## 📊 Vérifier la Conversion

```matlab
%% Test rapide
data = load('lstm_model.mat');

% Vérifier formes
input_shape = data.input_shape
output_shape = data.output_shape

% Vérifier poids
lstm_w0 = data.lstm_1_w0;
disp(['Poids LSTM: ' mat2str(size(lstm_w0))]);

dense_w0 = data.dense_1_w0;
disp(['Poids Dense: ' mat2str(size(dense_w0))]);
```

---

## 📂 Fichiers Disponibles

| Fichier | Description |
|---------|-------------|
| `lstm_model.mat` | ✅ **Poids du modèle** (à utiliser en MATLAB) |
| `lstm_acceleration_model.h5` | Modèle original (Keras) |
| `convert_simple.py` | Script de conversion (non-interactif) |
| `convert_h5_to_mat.py` | Script avec plus d'options |
| `H5_TO_MAT_SUMMARY.md` | Guide complet |
| `CONVERSION_H5_TO_MAT.md` | Documentation détaillée |

---

## 🔄 Si Vous Voulez Reconvertir

```bash
# Option 1: Rapide (SciPy)
python convert_simple.py

# Option 2: Menu interactif (plus d'options)
python convert_h5_to_mat.py
```

Choisir:
```
1. SciPy .mat (recommandé) ✅
2. NPZ (compression)
3. ONNX (Simulink)
4. Export détaillé (production)
5. Toutes les méthodes
```

---

## ⚡ Prochaines Étapes

### Pour MATLAB pur:
```matlab
data = load('lstm_model.mat');
% Implémenter forward LSTM manuellement (voir guide)
```

### Pour MATLAB + Deep Learning Toolbox:
```matlab
net = importKerasNetwork('lstm_acceleration_model.h5');
a = predict(net, u_normalized);
```

### Pour Simulink:
```
1. Générer ONNX: python convert_h5_to_mat.py → option 3
2. Ouvrir Simulink
3. Add-Ons → Deep Learning ONNX Importer
4. Charger lstm_model.onnx
5. ✓ Bloc auto-généré!
```

---

## 💡 Conseils

**✅ Utiliser:** Deep Learning Toolbox (easiest)  
**⚠️ Si pas d'accès:** Fichier .mat + implémenter manuellement  
**🎯 Pour Simulink:** Exporter en ONNX

---

## 🆘 Problèmes?

**Q: Erreur "Could not convert None"**  
A: Corriger les formes d'entrée (None = batch dimension). ✅ DÉJÀ FAIT!

**Q: Comment utiliser dans contrôleur?**  
A: Voir section "Pour Votre Contrôleur Drone"

**Q: Peut-on modifier les poids?**  
A: Oui, ce sont juste des matrices MATLAB

**Q: Taille fichier?**  
A: ~0 KB (très petit, 329 paramètres seulement!)

---

## 📖 Documentation Complète

- **H5_TO_MAT_SUMMARY.md** → Résumé rapide (cette page)
- **CONVERSION_H5_TO_MAT.md** → Guide détaillé (4 méthodes)
- **load_lstm_model.m** → Utilitaires MATLAB

---

## ✨ Résumé

```
H5 (Keras)
    ↓ python convert_simple.py
MAT (MATLAB) ✅
    ↓ load('lstm_model.mat')
Poids dans MATLAB
    ↓ Utiliser dans contrôleur
Prédictions d'accélération
```

**🎉 Vous êtes prêt!**

---

**Prochaines commandes:**

```matlab
% MATLAB
data = load('lstm_model.mat');
net = importKerasNetwork('lstm_acceleration_model.h5');
```

```bash
# Python (si besoin de reconvertir)
python convert_h5_to_mat.py
```

**Questions?** Voir `CONVERSION_H5_TO_MAT.md`
