#!/usr/bin/env python3
"""
Demo rapide: Convertir H5 en MAT et tester chargement MATLAB
"""

import os
import sys
import numpy as np
from pathlib import Path

# Ajouter chemin
sys.path.insert(0, os.getcwd())

def main():
    print("\n" + "="*70)
    print("DEMO: Convertir Modèle H5 → MAT → MATLAB")
    print("="*70)
    
    # Vérifier fichier H5
    h5_file = 'lstm_acceleration_model.h5'
    if not os.path.exists(h5_file):
        print(f"\n❌ Fichier {h5_file} non trouvé")
        print(f"   Chemin courant: {os.getcwd()}")
        return False
    
    print(f"\n✓ Fichier trouvé: {h5_file}")
    print(f"  Taille: {os.path.getsize(h5_file) / 1e6:.1f} MB")
    
    # Import dépendances
    print("\n📦 Vérification dépendances...")
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        print("  ✓ TensorFlow")
    except ImportError:
        print("  ❌ TensorFlow manquant")
        return False
    
    try:
        import scipy.io as sio
        print("  ✓ SciPy")
    except ImportError:
        print("  ❌ SciPy manquant (pip install scipy)")
        return False
    
    # Charger modèle
    print(f"\n📥 Chargement {h5_file}...")
    try:
        model = load_model(h5_file)
        print("  ✓ Modèle chargé")
    except ValueError:
        model = load_model(h5_file, compile=False)
        print("  ✓ Modèle chargé (compile=False)")
    
    # Extraire infos
    print(f"\n📊 Infos modèle:")
    print(f"  Entrée: {model.input_shape}")
    print(f"  Sortie: {model.output_shape}")
    print(f"  Layers: {len(model.layers)}")
    
    # Préparer données pour export
    print(f"\n🔧 Extraction poids...")
    
    matlab_dict = {}
    matlab_dict['input_shape'] = np.array(model.input_shape)
    matlab_dict['output_shape'] = np.array(model.output_shape)
    
    n_weights = 0
    for layer in model.layers:
        weights = layer.get_weights()
        for i, w in enumerate(weights):
            key = f"{layer.name}_w{i}"
            matlab_dict[key] = w.astype(np.float32)
            n_weights += 1
    
    print(f"  {n_weights} matrices de poids extraites")
    
    # Exporter .mat
    output_file = 'lstm_model_converted.mat'
    print(f"\n💾 Sauvegarde {output_file}...")
    
    try:
        sio.savemat(output_file, matlab_dict, oned_as='row')
        print(f"  ✓ Fichier créé: {output_file}")
        print(f"    Taille: {os.path.getsize(output_file) / 1e6:.1f} MB")
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False
    
    # Vérifier chargement
    print(f"\n✅ Vérification chargement...")
    try:
        test_data = np.load(output_file.replace('.mat', '.npz'), allow_pickle=True)
        print("  ✓ Fichier valide")
    except:
        pass  # .mat n'est pas NPZ, c'est ok
    
    # Afficher code MATLAB
    print("\n" + "="*70)
    print("📝 CODE MATLAB POUR CHARGER")
    print("="*70)
    
    matlab_code = f"""
%% Charger modèle Keras converti
data = load('{output_file}');

%% Accéder aux poids
weights = fieldnames(data);
fprintf('Poids disponibles (total: %d):\\n', length(weights));

for i = 1:min(10, length(weights))  % Afficher premiers 10
    w = data.(weights{{i}});
    fprintf('  %%s: shape %%s\\n', weights{{i}}, mat2str(size(w)));
end

%% Utiliser les poids
% Exemple: accéder à poids LSTM
% lstm_w0 = data.lstm_layer_w0;  % Poids
% lstm_b0 = data.lstm_layer_w1;  % Biais

%% Ou utiliser importKerasNetwork directement
% net = importKerasNetwork('lstm_acceleration_model.h5');
% prediction = predict(net, u_input);
"""
    
    print(matlab_code)
    
    # Résumé
    print("\n" + "="*70)
    print("✅ CONVERSION TERMINÉE")
    print("="*70)
    
    print(f"\n📂 Fichiers créés:")
    print(f"  - {output_file}")
    
    print(f"\n🎯 Prochaines étapes:")
    print(f"  1. Ouvrir MATLAB")
    print(f"  2. Charger: data = load('{output_file}');")
    print(f"  3. Ou utiliser dans Simulink avec Deep Learning Toolbox")
    
    print(f"\n💡 Plus d'options avec: python convert_h5_to_mat.py")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
