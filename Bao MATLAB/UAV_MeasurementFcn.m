function yk = UAV_MeasurementFcn(xk, uk)
% UAV_MeasurementFcn – fonction de mesure pour le NL-MPC
%   Ici on choisit de renvoyer la position z uniquement.
%
%   xk : état actuel = [h (16x1); z; v_z]
%   yk : sortie mesurée pour le MPC (ici : position z)

    nx_rnn = 16;

    z  = xk(nx_rnn+1);   % position
    % vz = xk(nx_rnn+2); % dispo si besoin plus tard

    yk = z;              % ny = 1
end
