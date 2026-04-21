function [acc,u,est,g,G_ts,phi_A_ts,phi_B_ts] = func_main_sim_1_antTrail_acc_T(...
    sw, num_agents, T, dt, omega_g, noise_sigma, T_half_period, ...
    k_trail, a_trail, rho, q_good, q_bad)
% Ant trail recruitment model
%
% Mapping to old variables:
%   g(:,t)   : private continuous noisy cue
%   est(:,t) : signed decision tendency = 2*P(A)-1, in [-1,1]
%   m(:,t)   : binary message / action, here simply equal to u(:,t)
%   u(:,t)   : binary decision (+1 for path A, -1 for path B)
%   acc      : instantaneous correctness
%   G_ts     : true environment state (+1 or -1)
%
% New ant-specific outputs:
%   phi_A_ts : pheromone level on path A
%   phi_B_ts : pheromone level on path B
%
% Core rule (scheme A):
%   effective_A = ... phi_A + ... g
%   effective_B = ...
%   P(A) = (k + effective_A)^a / [ (k + effective_A)^a + (k + effective_B)^a ]
%   Deneubourg/Beckers choice law
%
% Pheromone update:
%   phi_A(t+1) = (1-rho)*phi_A(t) + sum_{i choosing A} q_i
%   phi_B(t+1) = (1-rho)*phi_B(t) + sum_{i choosing B} q_i
%   q_i = q_good if chosen path matches G_t, else q_bad
%
% Notes:
%   - network and diz_list are kept only for interface compatibility.
%   - this is a global shared-field (stigmergy) model, so network is unused.
%   - m is already binary, so diz_list is unused.

len_time_steps = round(T / dt);

% preallocate
g = zeros(num_agents, len_time_steps);
est = zeros(num_agents, len_time_steps);
u = zeros(num_agents, len_time_steps);
acc = zeros(num_agents, len_time_steps - 1);
G_ts = zeros(len_time_steps, 1);

phi_A_ts = zeros(len_time_steps, 1);
phi_B_ts = zeros(len_time_steps, 1);

% ---- environment G(t): square wave flipping every T_half_period ----
G = @(t, half_period, high_value, low_value) ...
    (mod(t, 2*half_period) < half_period) * high_value + ...
    (mod(t, 2*half_period) >= half_period) * low_value;

for t = 1:len_time_steps
    G_t = G((t-1)*dt, T_half_period, 1.0, -1.0);
    G_ts(t) = G_t;
end

% randomize initial global sign
%if rand < 0.5
%    G_ts = -G_ts;
%end

% ---- initialize ----
% initial private cue
g(:,1) = noise_sigma * randn(num_agents,1);

% initial random actions/messages
u(:,1) = 2 * (rand(num_agents,1) > 0.5) - 1;

% initial pheromone: start from zero
phi_A_ts(1) = 0;
phi_B_ts(1) = 0;

% initial signed tendency: neutral
est(:,1) = 0;

% drift function for private cue
drift = @(g_i, G_t) -omega_g * (g_i - G_t);

for t = 2:len_time_steps

    % ---------- update private cue g by Heun ----------
    G_t_prev = G_ts(t-1);
    G_t_now  = G_ts(t);

    dW = randn(num_agents, 1) * sqrt(dt);

    drift_prev = drift(g(:,t-1), G_t_prev);
    g_pred     = g(:,t-1) + drift_prev * dt + noise_sigma * dW;
    drift_pred = drift(g_pred, G_t_now);

    g(:,t) = g(:,t-1) + 0.5 * (drift_prev + drift_pred) * dt + noise_sigma * dW;

    % binary private observation o_i in {-1,+1}
    %o = sign(g(:,t));
    %z0 = (o == 0);
    %if any(z0)
    %    o(z0) = 1;
    %end

    % ---------- ant trail recruitment choice rule ----------
    % scheme A:
    %   effective_A = sw * phi_A + mw * 1{o=+1}
    %   effective_B = sw * phi_B + mw * 1{o=-1}

    %eff_A = sw * phi_A_ts(t-1) + mw * (o == 1);
    %eff_B = sw * phi_B_ts(t-1) + mw * (o == -1);

    g_pos = max(g(:,t), 0);
    g_neg = max(-g(:,t), 0);    
    eff_A = sw/(1+sw) * phi_A_ts(t-1) + 1/(1+sw) * g_pos;
    eff_B = sw/(1+sw) * phi_B_ts(t-1) + 1/(1+sw) * g_neg;

    %eff_A = sw/(1+sw) * phi_A_ts(t-1) + 1/(1+sw) * g(:,t);
    %eff_B = sw/(1+sw) * phi_B_ts(t-1) + 1/(1+sw) * g(:,t);
    %eff_A = max(eff_A, 0);
    %eff_B = max(eff_B, 0);

    num_A = (k_trail + eff_A) .^ a_trail;
    num_B = (k_trail + eff_B) .^ a_trail;

    P_A = num_A ./ (num_A + num_B);

    % store signed tendency in [-1,1]
    est(:,t) = 2 * P_A - 1;

    % sample binary action
    u(:,t) = 2 * (rand(num_agents,1) < P_A) - 1;


    % diz_list kept only for compatibility; m is already binary
    % if you really want, you could discretize est here, but not needed

    % ---------- pheromone deposition ----------
    % agents choosing the correct path deposit q_good; otherwise q_bad
    q = q_bad * ones(num_agents,1);
    q(u(:,t) == G_t_now) = q_good;

    dep_A = sum(q(u(:,t) == 1));
    dep_B = sum(q(u(:,t) == -1));

    % evaporation + deposition
    phi_A_ts(t) = (1 - rho) * phi_A_ts(t-1) + dep_A;
    phi_B_ts(t) = (1 - rho) * phi_B_ts(t-1) + dep_B;

    % ---------- accuracy ----------
    acc(:,t-1) = 1.0 - abs(G_t_now - u(:,t)) / 2.0;

end

end