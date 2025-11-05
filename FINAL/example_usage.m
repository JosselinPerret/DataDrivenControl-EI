%% SIMPLE ALTITUDE CONTROLLER - EXAMPLE USAGE
% This script demonstrates how to use the DroneController class
%
% Requirements:
% - MATLAB R2020b or newer
% - Deep Learning Toolbox
% - lstm_acceleration_model.h5 (Keras model)
% - bdd_in_mat_05.csv (training data for normalization)

clear all; close all; clc;

fprintf('========================================\n');
fprintf('Drone Altitude Controller - MATLAB\n');
fprintf('========================================\n\n');

%% Initialize Controller
model_path = './lstm_acceleration_model.h5';

if ~isfile(model_path)
    fprintf('ERROR: Model file not found: %s\n', model_path);
    fprintf('Make sure lstm_acceleration_model.h5 is in the current directory\n');
    return;
end

% Create controller instance
controller = DroneController(model_path);

%% Test 1: Simple Hover Test
fprintf('\n========== TEST 1: HOVER TEST ==========\n');
fprintf('Testing: Can the model predict correct accelerations?\n\n');

fprintf('Control Input -> Predicted Acceleration\n');
fprintf('========================================\n');

for u_test = [-1.0, -0.5, 0.0, 0.5, 0.7, 1.0]
    a = controller.predict_acceleration(u_test);
    fprintf('u = %+.2f  ->  a = %+.2f m/s^2\n', u_test, a);
end

fprintf('\nExpected: u=0.7 should give a≈0 (hovering equilibrium)\n');

%% Test 2: Control for Different Heights
fprintf('\n========== TEST 2: ALTITUDE TRACKING ==========\n\n');

test_heights = [5, 10, 15];

for h_target = test_heights
    fprintf('-------------------------------------------\n');
    fprintf('Targeting height: %.1f m\n', h_target);
    fprintf('-------------------------------------------\n');
    
    % Reset controller
    controller.reset();
    controller.h_ref = h_target;
    
    % Simulate
    controller.simulate(30);  % 30 seconds
    
    % Print stats
    controller.print_stats();
    
    % Update plots
    controller.update_plots();
    
    % Pause between tests
    pause(2);
end

%% Test 3: Dynamic Reference Change
fprintf('\n========== TEST 3: DYNAMIC REFERENCE CHANGE ==========\n\n');

controller.reset();

% Simulate with changing reference
fprintf('Simulating with reference change at t=15s\n');
fprintf('Initial target: 10m, then change to 5m\n\n');

n_steps = round(30 / controller.DT);

for step = 1:n_steps
    % Change reference at halfway point
    if step > 150  % 150 steps * 0.05s = 7.5s ... actually at 150*0.05=7.5s
        controller.h_ref = 5.0;
    else
        controller.h_ref = 10.0;
    end
    
    % Compute control
    u_new = controller.compute_control(controller.h, controller.v, controller.h_ref);
    controller.u = 0.8 * controller.u + 0.2 * u_new;
    
    % Predict acceleration
    a = controller.predict_acceleration(controller.u);
    
    % Update state
    h_new = controller.h + controller.v * controller.DT + 0.5 * a * controller.DT^2;
    v_new = controller.v + a * controller.DT;
    
    % Safety
    if h_new < 0
        h_new = 0.0;
        v_new = max(v_new, 0.0);
    end
    
    controller.h = h_new;
    controller.v = v_new;
    
    % Store history
    controller.time_history = [controller.time_history; controller.current_time];
    controller.h_history = [controller.h_history; controller.h];
    controller.v_history = [controller.v_history; controller.v];
    controller.h_ref_history = [controller.h_ref_history; controller.h_ref];
    controller.u_history = [controller.u_history; controller.u];
    controller.a_history = [controller.a_history; a];
    
    controller.current_time = controller.current_time + controller.DT;
end

fprintf('Final height: %.3f m\n', controller.h);
fprintf('Final reference: %.1f m\n', controller.h_ref);
fprintf('Final error: %.3f m\n\n', controller.h_ref - controller.h);

controller.update_plots();
controller.print_stats();

fprintf('\n========================================\n');
fprintf('All tests completed!\n');
fprintf('========================================\n');
