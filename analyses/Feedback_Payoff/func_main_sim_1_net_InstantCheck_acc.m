function [acc,u,est,m,social_switch,g,G_ts] = func_main_sim_1_net_InstantCheck_acc(...
    sw, num_agents, T, dt, T_half_period, omega_g, noise_sigma,...
    network, diz_list, mw, check_rate)
len_time_steps = round(T / dt);

% simplified model:
% each agent checks current-step decision with probability check_rate
% if checked and wrong, flip social switch immediately

g = zeros(num_agents, len_time_steps);
est = zeros(num_agents, len_time_steps);
m = zeros(num_agents, len_time_steps); % messages
u = zeros(num_agents, len_time_steps); % decisions
acc = zeros(num_agents, len_time_steps - 1);
G_ts = zeros(len_time_steps,1);

social_switch = ones(num_agents, len_time_steps);
% 1: receiving social messages
% 0: shut down social inputs

% G(t) function
G = @(t, half_period, high_value, low_value) ...
    (mod(t, 2*half_period) < half_period) * high_value + ...
    (mod(t, 2*half_period) >= half_period) * low_value;
for t = 1:len_time_steps
    G_t = G((t-1)*dt, T_half_period, 1.0, -1.0);
    G_ts(t) = G_t;
end
if rand < 0.5
    G_ts = - G_ts;
end

% initial actions: random choice of -1 or 1
m(:,1) = 2 * (rand(num_agents, 1) > 0.5) - 1;

% precompute degree and use sparse network for speed
deg = sum(network, 1).';
if ~issparse(network), network = sparse(network); end

for t = 2:len_time_steps
    G_t      = G_ts(t-1);            % previous-step environment
    G_t_next = G_ts(t);              % current-step environment
    dW = randn(num_agents, 1) * sqrt(dt);

    % Heun predictor-corrector
    drift_prev = -omega_g * (g(:,t-1) - G_t);
    g_pred     = g(:,t-1) + drift_prev * dt + noise_sigma * dW;
    drift_pred = -omega_g * (g_pred   - G_t_next);
    g(:,t) = g(:,t-1) + 0.5*(drift_prev + drift_pred)*dt + noise_sigma * dW;

    sw_switched = sw .* social_switch(:,t-1);
    est(:,t) = ( sw_switched .* (network' * m(:,t-1)) + g(:,t) ) ./ ( sw_switched .* deg + 1 );
    m(:,t) = mw * est(:,t) + (1-mw) * g(:,t);

    u(:,t) = sign(est(:,t));
    z = (u(:,t) == 0);
    if any(z), u(z,t) = 1; end

    if ~isempty(diz_list)
        m(:,t) = func_discretize_to_list(m(:,t), diz_list);
    end

    acc(:,t-1) = 1.0 - abs(G_t - u(:,t)) / 2.0;

    social_switch(:,t) = social_switch(:,t-1);
    do_check = rand(num_agents, 1) < check_rate;
    is_wrong = (u(:,t) ~= G_t);
    flip = do_check & is_wrong;
    social_switch(flip,t) = 1 - social_switch(flip,t-1);
end

end
