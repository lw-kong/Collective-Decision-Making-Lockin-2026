function [acc,u,u_theta,g_theta,g,G_ts,iter_all] = func_main_sim_1_net_Ringv3ty(...
    decay_rate, num_agents, T, dt, omega_g, noise_sigma, network, ring_params)
len_time_steps = round(T / dt);

% sim_1
% now the initial G state is random from +1 or -1

g = zeros(num_agents, len_time_steps);
g_theta = zeros(num_agents, len_time_steps); % 0~2pi
u = zeros(num_agents, len_time_steps);
u_theta = zeros(num_agents, len_time_steps); % 0~2pi
acc = zeros(num_agents, len_time_steps - 1);
G_ts = zeros(len_time_steps,1);
iter_all = zeros(num_agents, len_time_steps);

% G(t) function
G = @(t, half_period, high_value, low_value) ...
    (mod(t, 2*half_period) < half_period) * high_value + ...
    (mod(t, 2*half_period) >= half_period) * low_value;
for t = 1:len_time_steps
    G_t = G((t-1)*dt, 250, 1.0, -1.0);
    G_ts(t) = G_t;
end
%if rand < 0.5
%    G_ts = - G_ts;
%end
% drift function
drift = @(g_i, G_t) -omega_g * (g_i - G_t);

% initial actions: random choice of -1 or 1
u_theta(:,1) = (2 * (rand(num_agents, 1) > 0.5) - 1) * pi/2;

g(:,1) = G_ts(1) * ones(num_agents,1);

for t = 2:len_time_steps
    G_t = G_ts(t-1);
    dW = randn(num_agents, 1) * sqrt(dt);

    for i = 1:num_agents
        % Predictor step
        g_pred = g(i, t-1) + drift(g(i, t-1), G_t) * dt + noise_sigma * dW(i);

        % Corrector step
        G_t_next = G_ts(t);
        g(i,t) = g(i, t-1) + ...
            0.5 * (drift(g(i, t-1), G_t) + drift(g_pred, G_t_next)) * dt + ...
            noise_sigma * dW(i);        
    end
    g_theta(:,t) = mod(pi*g(:,t)/2,2*pi);

    for i = 1:num_agents
        selected_indices = network(:,i)==1;
        %sum_n_est = sum(est(selected_indices, t-1)); % last time step

        % ring0
        input_social = [u_theta(selected_indices, t-1), ...
            decay_rate*ones(sum(selected_indices),1)];
        input_ob = [g_theta(i,t),1];
        input0 = [input_ob;input_social];
        [temp_0,~,iter_run] = func_ring_v3t(input0, ring_params);
        u_theta(i,t) = temp_0;
        iter_all(i,t) = iter_run;
    end
        
    u(:,t) = sin(u_theta(:,t)); 
    % action is a continuous variable, a direction
    % message = action in this case   
    acc(:, t-1) = 1.0 - abs(G_t - u(:,t)) / 2.0;

end
%acc_avg = mean(acc(:));
end
