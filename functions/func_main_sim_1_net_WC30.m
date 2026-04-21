function [acc,u,est,m,g,G_ts] = func_main_sim_1_net_WC30(...
    decay_rate, num_agents, T, dt, omega_g, noise_sigma,...
    network, diz_list, mw)
len_time_steps = round(T / dt);

% sim_1
% now the initial G state is random from +1 or -1

% _WC32
% weighted messages
% m = mw * e + (1-mw) * o

g = zeros(num_agents, len_time_steps);
est = zeros(num_agents, len_time_steps);
m = zeros(num_agents, len_time_steps); % messages
u = zeros(num_agents, len_time_steps); % decisions
acc = zeros(num_agents, len_time_steps - 1);
G_ts = zeros(len_time_steps,1);

% G(t) function
G = @(t, half_period, high_value, low_value) ...
    (mod(t, 2*half_period) < half_period) * high_value + ...
    (mod(t, 2*half_period) >= half_period) * low_value;
for t = 1:len_time_steps
    G_t = G((t-1)*dt, 250, 1.0, -1.0);
    G_ts(t) = G_t;
end
if rand < 0.5
    G_ts = - G_ts;
end
% drift function
drift = @(g_i, G_t) -omega_g * (g_i - G_t);

% initial actions: random choice of -1 or 1
u(:,1) = 2 * (rand(num_agents, 1) > 0.5) - 1;

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

    for i = 1:num_agents
        selected_indices = network(:,i)==1;
        %sum_n_est = sum(est(selected_indices, t-1)); % last time step
        est(i,t) = ( decay_rate * sum(m(selected_indices, t-1)) + g(i,t) )...
            / (decay_rate * sum(selected_indices) + 1) ;
        m(i,t) = mw * est(i,t) + (1-mw) * g(i,t);
        u(i,t) = sign( est(i,t) );        

        if ~isempty(diz_list)
            m(i,t) = func_discretize_to_list(m(i,t), diz_list); % WC12
        end

        acc(i, t-1) = 1.0 - abs(G_t - u(i,t)) / 2.0;
    end
end
%acc_avg = mean(acc(:));
end
