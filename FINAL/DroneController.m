%% SIMPLE ALTITUDE CONTROLLER - MATLAB VERSION
% Real-time drone altitude control with LSTM acceleration prediction
% Ported from simple_controller.py

classdef DroneController < handle
    %DRONECONTROLLER Main controller class
    
    properties
        % Configuration
        DT = 0.05;              % Sampling time (seconds) - 20 Hz
        G = 9.81;               % Gravity (m/s^2)
        GLOBAL_MAX_ABS_Y = 19.62;
        
        % LSTM Model
        lstm_model;             % Loaded neural network model
        scaler;                 % MinMaxScaler for input normalization
        n_timesteps;            % Number of input timesteps
        
        % Controller gains
        kp = 0.40;              % Proportional gain
        kd = 0.15;              % Derivative gain
        ki = 0.12;              % Integral gain
        u_hover = 0.70;         % Equilibrium control for hovering
        
        % State
        h = 0.0;                % Current height (m)
        v = 0.0;                % Current velocity (m/s)
        u = 0.0;                % Current control input
        h_ref = 5.0;            % Reference height
        integral_error = 0.0;   % Integral accumulator
        
        % History for plotting
        time_history = [];
        h_history = [];
        v_history = [];
        h_ref_history = [];
        u_history = [];
        a_history = [];
        current_time = 0.0;
        
        % GUI handles
        fig;
        ax_h;
        ax_v;
        ax_u;
        ax_a;
        
        % Control flags
        running = false;
        
        % Data scaling
        data_min;
        data_max;
    end
    
    methods
        function obj = DroneController(model_path)
            % Initialize controller
            
            fprintf('Initializing Drone Controller...\n');
            
            % Load LSTM model
            fprintf('Loading LSTM model: %s\n', model_path);
            try
                obj.lstm_model = importKerasNetwork(model_path, 'OutputLayerType', 'regression');
                obj.n_timesteps = obj.lstm_model.Layers(1).InputSize(1);
            catch ME
                fprintf('Error loading model: %s\n', ME.message);
                fprintf('Make sure you have Deep Learning Toolbox and model converter\n');
                return;
            end
            
            fprintf('Model loaded: input timesteps = %d\n', obj.n_timesteps);
            
            % Load training data for scaler normalization
            obj.load_training_data();
            
            % Initialize plots
            obj.create_plots();
        end
        
        function load_training_data(obj)
            % Load training data and fit scaler
            
            fprintf('Loading training data for normalization...\n');
            try
                data_in = readmatrix('../bdd_in_mat_05.csv');
                fprintf('Training data shape: %d x %d\n', size(data_in, 1), size(data_in, 2));
                fprintf('Training data range: [%.4f, %.4f]\n', min(data_in(:)), max(data_in(:)));
            catch
                fprintf('WARNING: Could not load bdd_in_mat_05.csv\n');
                fprintf('Using default range [-1, 1]\n');
                data_in = [-1; 1];
            end
            
            % MinMaxScaler: normalize to [min, max]
            obj.data_min = min(data_in(:));
            obj.data_max = max(data_in(:));
            
            fprintf('Scaler: min=%.4f, max=%.4f\n', obj.data_min, obj.data_max);
        end
        
        function create_plots(obj)
            % Create figure and subplots
            
            obj.fig = figure('Name', 'Drone Altitude Controller', 'NumberTitle', 'off');
            obj.fig.Position = [100 100 1200 800];
            
            % 4 subplots
            obj.ax_h = subplot(4, 1, 1);
            obj.ax_v = subplot(4, 1, 2);
            obj.ax_u = subplot(4, 1, 3);
            obj.ax_a = subplot(4, 1, 4);
            
            % Labels
            ylabel(obj.ax_h, 'Height (m)');
            ylabel(obj.ax_v, 'Velocity (m/s)');
            ylabel(obj.ax_u, 'Control Input');
            ylabel(obj.ax_a, 'Acceleration (m/s^2)');
            xlabel(obj.ax_a, 'Time (s)');
            
            % Grid
            grid(obj.ax_h, 'on');
            grid(obj.ax_v, 'on');
            grid(obj.ax_u, 'on');
            grid(obj.ax_a, 'on');
        end
        
        function a = predict_acceleration(obj, u_value)
            % Predict acceleration for constant control input
            %
            % Normalization:
            % 1. Normalize u with scaler fitted on training data
            % 2. Pass to LSTM model
            % 3. Denormalize: a = y_norm * GLOBAL_MAX_ABS_Y - G
            
            try
                % Create constant sequence
                u_seq = ones(obj.n_timesteps, 1) * u_value;
                
                % Normalize (MinMaxScaler)
                % Formula: (x - min) / (max - min) * (range_max - range_min) + range_min
                % For feature_range=(-1, 1): scale to [-1, 1]
                u_normalized = (u_seq - obj.data_min) / (obj.data_max - obj.data_min) * 2 - 1;
                
                % Reshape for LSTM: (batch, timesteps, features) = (1, n_timesteps, 1)
                u_reshaped = reshape(u_normalized, 1, obj.n_timesteps, 1);
                
                % Predict
                y_normalized = predict(obj.lstm_model, u_reshaped);
                
                % Get last timestep (LSTM outputs sequence)
                y_norm_last = y_normalized(1, end, 1);
                
                % Denormalize
                a = single(y_norm_last) * obj.GLOBAL_MAX_ABS_Y - obj.G;
                
            catch ME
                fprintf('Prediction error: %s\n', ME.message);
                a = -obj.G;  % Free fall on error
            end
        end
        
        function u = compute_control(obj, h, v, h_ref)
            % PID+Feedforward control
            % u = u_hover + kp*error + ki*integral(error) - kd*velocity
            
            error = h_ref - h;
            
            % Accumulate integral error
            obj.integral_error = obj.integral_error + error * obj.DT;
            
            % Anti-windup: limit integral
            obj.integral_error = max(min(obj.integral_error, 2.0), -2.0);
            
            % Compute control
            u = obj.u_hover;                           % Feed-forward
            u = u + obj.kp * error;                    % Proportional
            u = u + obj.ki * obj.integral_error;       % Integral
            u = u - obj.kd * v;                        % Derivative
            
            % Clamp to [-1, 1]
            u = max(min(u, 1.0), -1.0);
        end
        
        function reset(obj)
            % Reset controller and state
            
            obj.h = 0.0;
            obj.v = 0.0;
            obj.u = 0.0;
            obj.current_time = 0.0;
            obj.integral_error = 0.0;
            
            obj.time_history = [];
            obj.h_history = [];
            obj.v_history = [];
            obj.h_ref_history = [];
            obj.u_history = [];
            obj.a_history = [];
        end
        
        function simulate(obj, duration)
            % Run simulation for given duration (seconds)
            %
            % Example: obj.simulate(30)  % Simulate 30 seconds
            
            if obj.running
                fprintf('Already running!\n');
                return;
            end
            
            obj.running = true;
            n_steps = round(duration / obj.DT);
            
            fprintf('\n=== Starting Simulation ===\n');
            fprintf('Duration: %.1f seconds (%d steps)\n', duration, n_steps);
            fprintf('Reference height: %.1f m\n\n', obj.h_ref);
            fprintf('Step  Time    Height  Velocity  Control  Accel   Error\n');
            fprintf('----  ----    ------  --------  -------  -----   -----\n');
            
            for step = 1:n_steps
                % Compute control
                u_new = obj.compute_control(obj.h, obj.v, obj.h_ref);
                
                % Smoothing filter (80/20)
                obj.u = 0.8 * obj.u + 0.2 * u_new;
                
                % Predict acceleration
                a = obj.predict_acceleration(obj.u);
                
                % Update state (kinematic equations)
                h_new = obj.h + obj.v * obj.DT + 0.5 * a * obj.DT^2;
                v_new = obj.v + a * obj.DT;
                
                % Safety: prevent negative height
                if h_new < 0
                    h_new = 0.0;
                    v_new = max(v_new, 0.0);
                end
                
                obj.h = h_new;
                obj.v = v_new;
                
                % Store history
                obj.time_history = [obj.time_history; obj.current_time];
                obj.h_history = [obj.h_history; obj.h];
                obj.v_history = [obj.v_history; obj.v];
                obj.h_ref_history = [obj.h_ref_history; obj.h_ref];
                obj.u_history = [obj.u_history; obj.u];
                obj.a_history = [obj.a_history; a];
                
                obj.current_time = obj.current_time + obj.DT;
                
                % Print progress
                if mod(step, round(20)) == 0  % Every 1 second
                    error = obj.h_ref - obj.h;
                    fprintf('%4d  %.2f   %7.3f  %8.3f   %7.3f  %6.2f  %6.3f\n', ...
                        step, obj.current_time, obj.h, obj.v, obj.u, a, error);
                end
            end
            
            fprintf('\n=== Simulation Complete ===\n');
            fprintf('Final height: %.3f m (target: %.1f m)\n', obj.h, obj.h_ref);
            fprintf('Final error: %.3f m\n', obj.h_ref - obj.h);
            
            obj.running = false;
            obj.update_plots();
        end
        
        function update_plots(obj)
            % Update all plots
            
            if isempty(obj.time_history)
                return;
            end
            
            % Clear previous plots
            cla(obj.ax_h);
            cla(obj.ax_v);
            cla(obj.ax_u);
            cla(obj.ax_a);
            
            % Height plot
            plot(obj.ax_h, obj.time_history, obj.h_history, 'b-', 'LineWidth', 2);
            hold(obj.ax_h, 'on');
            plot(obj.ax_h, obj.time_history, obj.h_ref_history, 'r--', 'LineWidth', 2);
            hold(obj.ax_h, 'off');
            ylabel(obj.ax_h, 'Height (m)');
            legend(obj.ax_h, 'Height', 'Reference', 'Location', 'northwest');
            grid(obj.ax_h, 'on');
            
            % Velocity plot
            plot(obj.ax_v, obj.time_history, obj.v_history, 'g-', 'LineWidth', 2);
            ylabel(obj.ax_v, 'Velocity (m/s)');
            grid(obj.ax_v, 'on');
            
            % Control input plot
            plot(obj.ax_u, obj.time_history, obj.u_history, 'm-', 'LineWidth', 2);
            ylabel(obj.ax_u, 'Control Input');
            ylim(obj.ax_u, [-1.2 1.2]);
            grid(obj.ax_u, 'on');
            
            % Acceleration plot
            plot(obj.ax_a, obj.time_history, obj.a_history, 'c-', 'LineWidth', 2);
            hold(obj.ax_a, 'on');
            yline(obj.ax_a, 0, 'k--', 'LineWidth', 0.5);
            hold(obj.ax_a, 'off');
            ylabel(obj.ax_a, 'Acceleration (m/s^2)');
            xlabel(obj.ax_a, 'Time (s)');
            grid(obj.ax_a, 'on');
            
            drawnow;
        end
        
        function print_stats(obj)
            % Print simulation statistics
            
            fprintf('\n=== SIMULATION STATISTICS ===\n');
            fprintf('Duration: %.2f seconds\n', obj.current_time);
            fprintf('Max height: %.3f m\n', max(obj.h_history));
            fprintf('Min height: %.3f m\n', min(obj.h_history));
            fprintf('Final height: %.3f m\n', obj.h);
            fprintf('Final error: %.3f m\n', obj.h_ref - obj.h);
            
            if ~isempty(obj.a_history)
                fprintf('Max acceleration: %.2f m/s^2\n', max(obj.a_history));
                fprintf('Min acceleration: %.2f m/s^2\n', min(obj.a_history));
            end
            
            if ~isempty(obj.u_history)
                fprintf('Max control: %.3f\n', max(obj.u_history));
                fprintf('Min control: %.3f\n', min(obj.u_history));
            end
        end
    end
end
