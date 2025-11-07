% Chargement des paramètres du RNN

%% Chargement du modèle RNN (une seule fois)

% Temps d'échantillonage
Ts = 0.05;

% Dimensions du modèle RNN (UAV)
nx_nn = 16;          % dimension de l'état caché du RNN
nx_phy =2;
nx = nx_nn + nx_phy;

ny = 1;           % une sortie : accélération az
nu = 1;           % une commande : tension rotor u1

% Horizon
N  = 25;          % horizon de prédiction
Nu = 3;           % horizon de commande

nb_iterations_optim = 50;   % itérations max du solveur NLP

% Poids du MPC
W_MPC_y = 20;    % poids sur la sortie (accélération az)
W_MPC_u = 10;      % poids sur la commande

%% Définition du NL-MPC basé sur le RNN du quadricoptère

% Création de l'objet NL-MPC avec le modèle RNN
nlmpc_nn = nlmpc(nx, ny, nu);

nlmpc_nn.Ts                = Ts;
nlmpc_nn.PredictionHorizon = N;
nlmpc_nn.ControlHorizon    = Nu;

% Modèle pour le MPC : on utilise les fonctions UAV_* qu'on a définies

% Fonction de transition d'état : x(k+1) = UAV_StateTransitionFcn(x(k), u(k))
% Cette fonction charge en interne Wx, Wu, bxu, u_mean, u_std depuis rnn_quadrotor_model.mat
nlmpc_nn.Model.StateFcn = "UAV_StateTransitionFcn";

% Fonction de mesure / sortie : y(k) = UAV_MeasurementFcn(x(k))
% Cette fonction charge C, b_y, y_mean, y_std et renvoie l'accélération az dénormalisée
nlmpc_nn.Model.OutputFcn = "UAV_MeasurementFcn";

% Modèle discret (dynamique déjà discrète via le RNN)
nlmpc_nn.Model.IsContinuousTime = false;

% Conditions initiales
x0 = zeros(nx,1);    % état caché initial du RNN (au repos)
u0 = 0;              % commande initiale

% Validation des fonctions de modèle
% (vérifie que StateFcn et OutputFcn ont les bonnes dimensions)
validateFcns(nlmpc_nn, x0, u0, [], []);


%% Poids dans la fonction coût du NL-MPC

% Même syntaxe que ton script original : propriété "W"
nlmpc_nn.Weights.OutputVariables          = W_MPC_y;
nlmpc_nn.Weights.ManipulatedVariables     = W_MPC_u;
nlmpc_nn.Weights.ManipulatedVariablesRate = 0.1;



%% Contraintes sur la commande (tension rotor)

% À adapter en fonction de ton ESC / alim
nlmpc_nn.MV.Min     = -1;     % tension min (ex : 0 V)
nlmpc_nn.MV.Max     = 1;    % tension max (ex : 10 V)

% Contraintes sur l'output (z)
nlmpc_nn.OV.Min = -200;     % à adapter à ton graphique
nlmpc_nn.OV.Max =  200;

% Contraintes sur la variation de commande
nlmpc_nn.MV.RateMin = -0.1;   % Δu min par Ts (à ajuster)
nlmpc_nn.MV.RateMax =  0.1;   % Δu max par Ts

%% Paramètres du solveur NLP interne

nlmpc_nn.Optimization.SolverOptions.MaxIterations = nb_iterations_optim;
nlmpc_nn.Optimization.UseSuboptimalSolution = true;

%% Ouverture du modèle Simulink
open("EINT_system_simulation.slx")