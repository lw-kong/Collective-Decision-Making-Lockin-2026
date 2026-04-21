function [acc,u,est,m,g,G_ts] = func_main_sim_1_net_WC30_TopKm_T(...
    sw, num_agents, T, dt, omega_g, noise_sigma,...
    network, diz_list, mw, topk, half_T)
len_time_steps = round(T / dt);

% sim_1
% now the initial G state is random from +1 or -1

% _WC32
% weighted messages
% m = mw * e + (1-mw) * o

% _acc
% accelerated version

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
    G_t = G((t-1)*dt, half_T, 1.0, -1.0);
    G_ts(t) = G_t;
end
%if rand < 0.5
%    G_ts = - G_ts;
%end
% drift function
drift = @(g_i, G_t) -omega_g * (g_i - G_t);

% initial actions: random choice of -1 or 1
m(:,1) = 2 * (rand(num_agents, 1) > 0.5) - 1;
%m(:,1) = 0.5*randn(num_agents,1);
m(:,1) = zeros(num_agents,1);
%m(:,1) = 0.7*ones(num_agents,1);
% ---- 预计算度数与稀疏化（大图可显著提速）----
% est 的邻居求和： s(:,t) = network' * m(:,t)
% 注意你原来用的是 network(:,i)==1（对列求和），因此度是列和：
deg = sum(network, 1).';             % num_agents x 1
if ~issparse(network), network = sparse(network); end



for t = 2:len_time_steps
    G_t      = G_ts(t-1);            % 上一步环境
    G_t_next = G_ts(t);              % 本步环境
    dW = randn(num_agents, 1) * sqrt(dt);

    % Heun 预测-校正法（向量化）
    drift_prev = -omega_g * (g(:,t-1) - G_t);
    g_pred     = g(:,t-1) + drift_prev * dt + noise_sigma * dW;
    drift_pred = -omega_g * (g_pred   - G_t_next);
    g(:,t) = g(:,t-1) + 0.5*(drift_prev + drift_pred)*dt + noise_sigma * dW;

    %est(:,t) = ( sw .* (network' * m(:,t-1)) + g(:,t) ) ./ ( sw .* deg + 1 );

    
    att_mean = zeros(num_agents,1);
    att_count = zeros(num_agents,1);
    for i = 1:num_agents
        in_idx = network(:, i);      % j -> i
        temp_input = m(in_idx,t-1);
    
        k_eff = min(topk, numel(temp_input));
        [~, order] = sort(abs(temp_input), 'descend');
        sel = order(1:k_eff);
        
        if k_eff > 0
            att_mean(i) = mean( temp_input(sel));
            att_count(i) = k_eff;
        end
    end

    est(:,t) = ( sw * att_count .* att_mean + g(:,t) ) ./ ( sw .* att_count + 1 );

    m(:,t) = mw * est(:,t) + (1-mw) * g(:,t);

    u(:,t) = sign(est(:,t));
    z = (u(:,t) == 0);
    if any(z), u(z,t) = 1; end

    if ~isempty(diz_list)
        m(:,t) = func_discretize_to_list(m(:,t), diz_list); % WC12
    end

    acc(:,t-1) = 1.0 - abs(G_t - u(:,t)) / 2.0; % does this really make sense?

end

end
