#!/usr/bin/env python3
"""
Convertir modèle Keras H5 en format MATLAB .mat

Méthodes:
1. Export poids + architecture (recommandé pour Simulink)
2. Export structure complète en JSON + NPZ
3. Export ONNX (si disponible)
"""

import numpy as np
import os
import sys
import json
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠ TensorFlow non disponible. Installez: pip install tensorflow")

try:
    import scipy.io as sio
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠ SciPy non disponible. Installez: pip install scipy")

try:
    import onnx
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def load_keras_model(model_path):
    """Charger modèle Keras avec fallback"""
    print(f"\n📦 Chargement du modèle: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Fichier non trouvé: {model_path}")
        return None
    
    try:
        model = load_model(model_path)
        print(f"✓ Modèle chargé avec compile=True")
    except ValueError as e:
        print(f"⚠ Erreur de chargement, essai avec compile=False...")
        try:
            model = load_model(model_path, compile=False)
            print(f"✓ Modèle chargé avec compile=False")
        except Exception as e2:
            print(f"❌ Erreur: {e2}")
            return None
    
    # Afficher infos
    print(f"\nArchitecture du modèle:")
    model.summary()
    
    return model


def extract_weights_and_architecture(model):
    """Extraire poids et architecture du modèle"""
    
    weights_dict = {}
    architecture = {}
    
    print(f"\n📋 Extraction des poids et architecture...")
    
    # Couches et poids
    for i, layer in enumerate(model.layers):
        layer_name = layer.name
        layer_weights = layer.get_weights()
        
        print(f"  Layer {i}: {layer_name} ({layer.__class__.__name__})")
        
        architecture[layer_name] = {
            'type': layer.__class__.__name__,
            'config': layer.get_config(),
            'n_weights': len(layer_weights),
            'weight_shapes': [w.shape for w in layer_weights]
        }
        
        # Stocker les poids
        for j, w in enumerate(layer_weights):
            key = f"{layer_name}_weight_{j}"
            weights_dict[key] = w
            print(f"    - Weight {j}: shape {w.shape}, dtype {w.dtype}")
    
    # Forme entrée/sortie
    input_shape = model.input_shape
    output_shape = model.output_shape
    
    print(f"\n📐 Forme d'entrée: {input_shape}")
    print(f"📐 Forme de sortie: {output_shape}")
    
    return weights_dict, architecture, input_shape, output_shape


def method_1_scipy_mat(model, output_path='lstm_model.mat'):
    """
    MÉTHODE 1: Exporter en .mat avec SciPy (RECOMMANDÉ pour MATLAB)
    
    Avantages:
    - Compatible directement avec MATLAB
    - Simples arrays à charger
    - Peut utiliser matconv.m dans MATLAB
    """
    
    print(f"\n{'='*60}")
    print("MÉTHODE 1: Export SciPy .mat (Recommandé)")
    print(f"{'='*60}")
    
    if not SCIPY_AVAILABLE:
        print("❌ SciPy non disponible")
        return False
    
    weights_dict, architecture, input_shape, output_shape = extract_weights_and_architecture(model)
    
    # Préparer dictionnaire pour MATLAB
    matlab_dict = {}
    
    # Ajouter infos architecture (sans None)
    matlab_dict['model_type'] = np.array([model.__class__.__name__], dtype=object)
    
    # Nettoyer les formes (remplacer None par -1 ou 0)
    input_shape_clean = tuple(s if s is not None else -1 for s in input_shape)
    output_shape_clean = tuple(s if s is not None else -1 for s in output_shape)
    
    matlab_dict['input_shape'] = np.array(input_shape_clean, dtype=np.float32)
    matlab_dict['output_shape'] = np.array(output_shape_clean, dtype=np.float32)
    
    # Ajouter poids
    for key, weight in weights_dict.items():
        matlab_dict[key] = weight.astype(np.float32)
    
    # Sauvegarder
    try:
        sio.savemat(output_path, matlab_dict, oned_as='row')
        print(f"\n✅ Modèle sauvegardé: {output_path}")
        print(f"   Taille: {os.path.getsize(output_path) / 1e6:.1f} MB")
        return True
    except Exception as e:
        print(f"❌ Erreur sauvegarde: {e}")
        return False


def method_2_h5_to_npz(model_path, output_path='lstm_model.npz'):
    """
    MÉTHODE 2: Copier H5 → NPZ (pour archivage Python)
    
    Avantages:
    - Compression meilleure que H5
    - Facile à charger en Python après
    - Formats multiples supportés
    """
    
    print(f"\n{'='*60}")
    print("MÉTHODE 2: H5 → NPZ")
    print(f"{'='*60}")
    
    # Charger le H5 directement avec h5py
    try:
        import h5py
    except ImportError:
        print("❌ h5py non disponible. Installez: pip install h5py")
        return False
    
    try:
        with h5py.File(model_path, 'r') as h5_file:
            npz_dict = {}
            
            def extract_from_h5(name, obj):
                if isinstance(obj, h5py.Dataset):
                    npz_dict[name] = np.array(obj)
            
            h5_file.visititems(extract_from_h5)
        
        # Sauvegarder en NPZ
        np.savez_compressed(output_path, **npz_dict)
        print(f"✅ NPZ créé: {output_path}")
        print(f"   Taille: {os.path.getsize(output_path) / 1e6:.1f} MB")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def method_3_onnx_conversion(model_path, output_path='lstm_model.onnx'):
    """
    MÉTHODE 3: Keras → ONNX (compatible avec C#, C++, JS)
    
    Avantages:
    - Format universel
    - Peut être importé dans Simulink via ONNX Import Tool
    - Meilleure compatibilité cross-platform
    """
    
    print(f"\n{'='*60}")
    print("MÉTHODE 3: Keras → ONNX")
    print(f"{'='*60}")
    
    try:
        import tf2onnx
        import onnx
    except ImportError:
        print("❌ tf2onnx ou onnx non disponible")
        print("   Installez: pip install tf2onnx onnx")
        return False
    
    try:
        # Charger modèle TensorFlow
        model = load_model(model_path, compile=False)
        
        # Spécifier input spec
        spec = (tf.TensorSpec((None, model.input_shape[1], 1), tf.float32, name="input"),)
        
        # Convertir
        output_path_onnx, _ = tf2onnx.convert.from_keras(model, input_signature=spec, 
                                                         output_path=output_path)
        
        print(f"✅ ONNX créé: {output_path}")
        print(f"   Taille: {os.path.getsize(output_path) / 1e6:.1f} MB")
        return True
        
    except Exception as e:
        print(f"❌ Erreur conversion ONNX: {e}")
        return False


def method_4_detailed_export(model, output_dir='lstm_export'):
    """
    MÉTHODE 4: Export détaillé (tous les fichiers)
    
    Crée un dossier avec:
    - architecture.json : structure du modèle
    - weights/ : dossier avec poids individuels
    - config.json : configuration d'inférence
    """
    
    print(f"\n{'='*60}")
    print("MÉTHODE 4: Export détaillé (Recommandé pour production)")
    print(f"{'='*60}")
    
    # Créer dossier
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/weights', exist_ok=True)
    
    # Architecture
    weights_dict, architecture, input_shape, output_shape = extract_weights_and_architecture(model)
    
    config = {
        'model_type': str(model.__class__.__name__),
        'input_shape': input_shape,
        'output_shape': output_shape,
        'num_layers': len(model.layers),
        'layers': {}
    }
    
    # Sauvegarder architecture en JSON
    for layer_name, layer_info in architecture.items():
        config['layers'][layer_name] = {
            'type': layer_info['type'],
            'n_weights': layer_info['n_weights'],
            'weight_shapes': [list(s) for s in layer_info['weight_shapes']]
        }
    
    with open(f'{output_dir}/architecture.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Architecture sauvegardée: {output_dir}/architecture.json")
    
    # Sauvegarder poids individuels
    for key, weight in weights_dict.items():
        filepath = f'{output_dir}/weights/{key}.npy'
        np.save(filepath, weight.astype(np.float32))
    
    print(f"✓ Poids sauvegardés: {output_dir}/weights/ ({len(weights_dict)} fichiers)")
    
    # Configuration pour inférence
    inference_config = {
        'input_shape': input_shape,
        'output_shape': output_shape,
        'input_dtype': 'float32',
        'output_dtype': 'float32',
        'normalization': {
            'enabled': True,
            'input_min': -1.0,
            'input_max': 1.0
        }
    }
    
    with open(f'{output_dir}/inference_config.json', 'w') as f:
        json.dump(inference_config, f, indent=2)
    
    print(f"✓ Config sauvegardée: {output_dir}/inference_config.json")
    print(f"\n✅ Export complet dans: {output_dir}/")
    return True


def main():
    """Main"""
    
    print("\n" + "="*60)
    print("CONVERTIR MODÈLE KERAS H5 EN FORMATS MATLAB")
    print("="*60)
    
    # Fichier modèle
    model_path = './lstm_acceleration_model.h5'
    
    # Vérifier existence
    if not os.path.exists(model_path):
        print(f"\n❌ Fichier non trouvé: {model_path}")
        print(f"   Chemin actuel: {os.getcwd()}")
        sys.exit(1)
    
    # Charger modèle
    model = load_keras_model(model_path)
    if model is None:
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("CHOISIR MÉTHODE DE CONVERSION")
    print(f"{'='*60}")
    print("\n1. SciPy .mat (Recommandé pour MATLAB) ✅")
    print("2. H5 → NPZ (Compression)")
    print("3. Keras → ONNX (Format universel)")
    print("4. Export détaillé (Production)")
    print("5. Toutes les méthodes")
    print("0. Quitter")
    
    choice = input("\nChoisir (0-5): ").strip()
    
    if choice == '1':
        method_1_scipy_mat(model, 'lstm_model.mat')
    
    elif choice == '2':
        method_2_h5_to_npz(model_path, 'lstm_model.npz')
    
    elif choice == '3':
        method_3_onnx_conversion(model_path, 'lstm_model.onnx')
    
    elif choice == '4':
        method_4_detailed_export(model, 'lstm_export')
    
    elif choice == '5':
        print("\n⏳ Exécution de toutes les méthodes...\n")
        method_1_scipy_mat(model, 'lstm_model.mat')
        method_2_h5_to_npz(model_path, 'lstm_model.npz')
        method_3_onnx_conversion(model_path, 'lstm_model.onnx')
        method_4_detailed_export(model, 'lstm_export')
        
        print(f"\n{'='*60}")
        print("✅ TOUTES LES CONVERSIONS TERMINÉES")
        print(f"{'='*60}")
        print("\nFichiers créés:")
        print("  - lstm_model.mat (SciPy)")
        print("  - lstm_model.npz (NumPy compressed)")
        print("  - lstm_model.onnx (ONNX)")
        print("  - lstm_export/ (Détaillé)")
    
    else:
        print("Annulé")
        return
    
    print(f"\n{'='*60}")
    print("📝 COMMENT UTILISER EN MATLAB")
    print(f"{'='*60}")
    
    print("\n**Pour charger les poids en MATLAB:**\n")
    print("```matlab")
    print("% Charger fichier .mat")
    print("data = load('lstm_model.mat');")
    print("\n% Accéder aux poids")
    print("weights_names = fieldnames(data);")
    print("for i = 1:length(weights_names)")
    print("    W = data.(weights_names{i});")
    print("    disp(['Poids: ' weights_names{i} ', shape: ' mat2str(size(W))]);")
    print("end")
    print("```")
    
    print("\n**Pour importer directement dans Simulink:**\n")
    print("1. Ouvrir Simulink")
    print("2. Simulink → Add-Ons → Deep Learning ONNX Importer")
    print("3. Charger: lstm_model.onnx")
    print("4. Générer bloc Simulink automatiquement")


if __name__ == '__main__':
    main()
