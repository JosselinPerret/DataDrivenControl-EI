#!/usr/bin/env python3
"""
Convertir H5 → MAT (version NON-INTERACTIVE)
Exécute automatiquement la conversion SciPy
"""

import os
import sys
import numpy as np

try:
    from tensorflow.keras.models import load_model
    import scipy.io as sio
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    print("Installer: pip install tensorflow scipy")
    sys.exit(1)


def convert_h5_to_mat(h5_file='lstm_acceleration_model.h5', 
                      mat_file='lstm_model.mat'):
    """Convertir H5 → MAT en SciPy"""
    
    print("\n" + "="*70)
    print("🔄 CONVERSION KERAS H5 → MATLAB .mat")
    print("="*70)
    
    # Étape 1: Vérifier fichier
    if not os.path.exists(h5_file):
        print(f"\n❌ Fichier non trouvé: {h5_file}")
        print(f"   Chemin courant: {os.getcwd()}")
        return False
    
    print(f"\n✓ Fichier trouvé: {h5_file}")
    size_mb = os.path.getsize(h5_file) / 1e6
    print(f"  Taille: {size_mb:.1f} MB")
    
    # Étape 2: Charger modèle
    print(f"\n📥 Chargement modèle...")
    try:
        model = load_model(h5_file)
    except ValueError:
        model = load_model(h5_file, compile=False)
    
    print(f"  ✓ Modèle chargé")
    print(f"    Input shape: {model.input_shape}")
    print(f"    Output shape: {model.output_shape}")
    
    # Étape 3: Extraire poids
    print(f"\n🔧 Extraction des poids...")
    
    matlab_dict = {}
    
    # Infos architecture (nettoyer les None)
    input_shape_clean = tuple(s if s is not None else -1 for s in model.input_shape)
    output_shape_clean = tuple(s if s is not None else -1 for s in model.output_shape)
    
    matlab_dict['input_shape'] = np.array(input_shape_clean, dtype=np.float32)
    matlab_dict['output_shape'] = np.array(output_shape_clean, dtype=np.float32)
    matlab_dict['num_layers'] = np.array([len(model.layers)], dtype=np.float32)
    
    # Poids
    n_weights = 0
    for layer in model.layers:
        layer_weights = layer.get_weights()
        
        for i, w in enumerate(layer_weights):
            key = f"{layer.name}_w{i}"
            matlab_dict[key] = w.astype(np.float32)
            n_weights += 1
            print(f"  {key}: shape {w.shape}")
    
    print(f"  ✓ {n_weights} matrices extraites")
    
    # Étape 4: Sauvegarder .mat
    print(f"\n💾 Sauvegarde du fichier .mat...")
    
    try:
        sio.savemat(mat_file, matlab_dict, oned_as='row')
        size_mb_out = os.path.getsize(mat_file) / 1e6
        print(f"  ✓ Fichier créé: {mat_file}")
        print(f"    Taille: {size_mb_out:.1f} MB")
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False
    
    return True


def show_matlab_code():
    """Afficher code MATLAB pour charger"""
    
    code = """
% ===== MATLAB =====
% Charger fichier .mat généré

clear; clc;

%% Charger poids
data = load('lstm_model.mat');

%% Afficher contenu
fprintf('Contenu du fichier:\\n');
weights_names = fieldnames(data);
for i = 1:length(weights_names)
    name = weights_names{i};
    w = data.(name);
    if isnumeric(w)
        fprintf('  %s: shape %s\\n', name, mat2str(size(w)));
    end
end

%% Accéder aux poids spécifiques
% lstm_weights_0 = data.lstm_1_w0;  % Poids LSTM
% lstm_weights_1 = data.lstm_1_w1;  % Récurrence
% lstm_bias = data.lstm_1_w2;       % Biais

% dense_weights = data.dense_1_w0;  % Poids Dense
% dense_bias = data.dense_1_w1;     % Biais Dense

%% Alternative: Utiliser Deep Learning Toolbox (recommandé)
% net = importKerasNetwork('lstm_acceleration_model.h5', ...
%     'OutputLayerType', 'regression');
% 
% % Prédiction
% u_input = randn(1, 2000, 1);
% y_pred = predict(net, u_input);
"""
    
    print("\n" + "="*70)
    print("📝 CODE MATLAB POUR CHARGER")
    print("="*70)
    print(code)


def main():
    
    print("\n" + "█"*70)
    print("█  CONVERTIR MODÈLE KERAS H5 → MATLAB MAT")
    print("█"*70)
    
    # Conversion
    success = convert_h5_to_mat('lstm_acceleration_model.h5', 'lstm_model.mat')
    
    if not success:
        sys.exit(1)
    
    # Code MATLAB
    show_matlab_code()
    
    # Résumé
    print("\n" + "="*70)
    print("✅ CONVERSION RÉUSSIE!")
    print("="*70)
    
    print(f"\n📂 Fichiers générés:")
    print(f"   - lstm_model.mat (poids du modèle)")
    
    print(f"\n🎯 Prochaines étapes:")
    print(f"   1. Ouvrir MATLAB")
    print(f"   2. Charger: data = load('lstm_model.mat');")
    print(f"   3. Utiliser poids pour inférence")
    
    print(f"\n💡 Documentation:")
    print(f"   - H5_TO_MAT_SUMMARY.md (résumé rapide)")
    print(f"   - CONVERSION_H5_TO_MAT.md (guide complet)")
    
    print(f"\n📚 Pour plus d'options:")
    print(f"   python convert_h5_to_mat.py (menu interactif)")
    
    print()


if __name__ == '__main__':
    main()
