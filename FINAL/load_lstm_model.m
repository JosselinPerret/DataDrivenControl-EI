%% Charger modèle LSTM depuis .mat exporté
% Ce script charge le modèle Keras converti en format MATLAB

function model_data = load_lstm_model(mat_file)
    %LOAD_LSTM_MODEL Charger modèle depuis fichier .mat
    %
    % Usage:
    %   model_data = load_lstm_model('lstm_model.mat');
    
    if nargin < 1
        mat_file = 'lstm_model.mat';
    end
    
    fprintf('\n📦 Chargement du modèle: %s\n', mat_file);
    
    % Vérifier existence
    if ~isfile(mat_file)
        error(['Fichier non trouvé: ' mat_file]);
    end
    
    % Charger fichier .mat
    data = load(mat_file);
    
    % Afficher contenu
    fprintf('\n📋 Contenu du fichier:\n');
    field_names = fieldnames(data);
    
    for i = 1:length(field_names)
        name = field_names{i};
        value = data.(name);
        
        if isnumeric(value)
            fprintf('   %s: shape %s, dtype %s\n', name, mat2str(size(value)), class(value));
        elseif iscell(value)
            fprintf('   %s: cell array, %d éléments\n', name, numel(value));
        else
            fprintf('   %s: %s\n', name, class(value));
        end
    end
    
    % Extraire infos
    model_data.weights_dict = data;
    
    if isfield(data, 'input_shape')
        model_data.input_shape = data.input_shape;
    else
        model_data.input_shape = [];
    end
    
    if isfield(data, 'output_shape')
        model_data.output_shape = data.output_shape;
    else
        model_data.output_shape = [];
    end
    
    fprintf('\n✓ Modèle chargé avec succès\n');
end


%% EXEMPLE D'UTILISATION
function demo_load_model()
    
    fprintf('\n=================================================================\n');
    fprintf('DÉMO: Charger et utiliser poids Keras en MATLAB\n');
    fprintf('=================================================================\n\n');
    
    % Charger modèle
    try
        model = load_lstm_model('lstm_model.mat');
    catch
        fprintf('❌ Fichier .mat non trouvé\n');
        fprintf('   Étape 1: Exécuter convert_h5_to_mat.py en Python\n');
        fprintf('   Étape 2: Charger avec load_lstm_model.m\n');
        return;
    end
    
    % Accéder aux poids
    fprintf('\n📊 Analyse des poids:\n');
    
    % Lister tous les poids
    weight_names = fieldnames(model.weights_dict);
    lstm_weights = {};
    dense_weights = {};
    
    for i = 1:length(weight_names)
        name = weight_names{i};
        if contains(name, 'lstm')
            lstm_weights{end+1} = name;
        elseif contains(name, 'dense')
            dense_weights{end+1} = name;
        end
    end
    
    fprintf('\nPoids LSTM:\n');
    for i = 1:length(lstm_weights)
        name = lstm_weights{i};
        w = model.weights_dict.(name);
        fprintf('   %s: shape %s\n', name, mat2str(size(w)));
    end
    
    fprintf('\nPoids Dense:\n');
    for i = 1:length(dense_weights)
        name = dense_weights{i};
        w = model.weights_dict.(name);
        fprintf('   %s: shape %s\n', name, mat2str(size(w)));
    end
    
    % Inférence manuelle (si souhaité)
    fprintf('\n⚙️  Prédiction manuelle:\n\n');
    
    % Créer entrée test
    n_timesteps = 2000;
    u_test = 0.5;  % Commande test
    
    % Séquence d''entrée (constant)
    u_sequence = ones(n_timesteps, 1) * u_test;
    
    fprintf('   Entrée: séquence de %d timesteps, u=%g\n', n_timesteps, u_test);
    
    % Normalisation (si nécessaire)
    u_normalized = 2 * (u_sequence - (-1)) / (1 - (-1)) - 1;  % Normalization
    
    % Forward pass LSTM (simplifié)
    % Note: C''est une approximation - pour inférence complète, utiliser importKerasNetwork
    fprintf('   Note: Pour inférence complète, utiliser importKerasNetwork()\n');
    
end


%% Alternative: Charger avec Deep Learning Toolbox
function model_imported = load_with_dlToolbox(h5_file)
    %LOAD_WITH_DLTOOLBOX Charger H5 directement avec importKerasNetwork
    %
    % Nécessite: Deep Learning Toolbox
    % Usage:
    %   model = load_with_dlToolbox('lstm_acceleration_model.h5');
    
    fprintf('\n🔧 Chargement avec importKerasNetwork...\n');
    
    try
        % Charger modèle H5 directement
        model_imported = importKerasNetwork(h5_file, 'OutputLayerType', 'regression');
        fprintf('✓ Modèle chargé avec Deep Learning Toolbox\n');
    catch ME
        fprintf('❌ Erreur: %s\n', ME.message);
        fprintf('   Deep Learning Toolbox nécessaire\n');
        model_imported = [];
    end
end


%% Afficher architecture (si fichier JSON disponible)
function display_architecture(json_file)
    
    if nargin < 1
        json_file = 'lstm_export/architecture.json';
    end
    
    if ~isfile(json_file)
        fprintf('❌ Fichier non trouvé: %s\n', json_file);
        return;
    end
    
    fprintf('\n📐 Architecture du modèle:\n');
    
    % Charger JSON
    json_text = fileread(json_file);
    json_data = jsondecode(json_text);
    
    fprintf('   Type: %s\n', json_data.model_type);
    fprintf('   Input shape: %s\n', mat2str(json_data.input_shape));
    fprintf('   Output shape: %s\n', mat2str(json_data.output_shape));
    fprintf('   Nombre de layers: %d\n', json_data.num_layers);
    
    fprintf('\n   Détail des couches:\n');
    layer_names = fieldnames(json_data.layers);
    
    for i = 1:length(layer_names)
        layer_name = layer_names{i};
        layer_info = json_data.layers.(layer_name);
        fprintf('      %d. %s (%s)\n', i, layer_name, layer_info.type);
        fprintf('         Poids: %d, shapes: ', layer_info.n_weights);
        for j = 1:length(layer_info.weight_shapes)
            fprintf('%s ', mat2str(layer_info.weight_shapes{j}));
        end
        fprintf('\n');
    end
end
