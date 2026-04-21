function [acc,u,est,g,G_ts,phi_A_ts,phi_B_ts] = func_main_sim_1_antTrail_acc_T_g( ...
    sw, g_input, k_trail, a_trail, rho, q_good, q_bad)
% Ant trail recruitment model with external private cue g(t)
%
% Inputs:
%   sw      : social weight between pheromone and private cue
%   g_input : external cue matrix, size [num_agents, len_time_steps]
%   k_trail, a_trail : Deneubourg/Beckers choice-law parameters
%   rho     : pheromone evaporation rate in [0,1]
%   q_good  : deposition amount if action matches sign(G_t)
%   q_bad   : deposition amount otherwise
%
% Outputs:
%   acc      : instantaneous correctness, size [num_agents, len_time_steps-1]
%   u        : binary decision (+1 for path A, -1 for path B)
%   est      : signed tendency in [-1,1], est = 2*P(A)-1
%   g        : equals g_input (for interface compatibility)
%   G_ts     : binary environment sign inferred from mean(g(:,t))
%   phi_A_ts : pheromone level on path A
%   phi_B_ts : pheromone level on path B

g = g_input;
[num_agents, len_time_steps] = size(g);

% infer binary "ground-truth" sign from external cue
G_ts = sign(mean(g, 1))';
z0 = (G_ts == 0);
if any(z0)
    G_ts(z0) = 1;
end

% preallocate
est = zeros(num_agents, len_time_steps);
u = zeros(num_agents, len_time_steps);
acc = zeros(num_agents, len_time_steps - 1);
phi_A_ts = zeros(len_time_steps, 1);
phi_B_ts = zeros(len_time_steps, 1);

% initialize
u(:,1) = 2 * (rand(num_agents,1) > 0.5) - 1;
est(:,1) = 0;
phi_A_ts(1) = 0;
phi_B_ts(1) = 0;

for t = 2:len_time_steps
    G_t_now = G_ts(t);

    % ant trail recruitment choice rule
    g_pos = max(g(:,t), 0);
    g_neg = max(-g(:,t), 0);

    eff_A = sw/(1+sw) * phi_A_ts(t-1) + 1/(1+sw) * g_pos;
    eff_B = sw/(1+sw) * phi_B_ts(t-1) + 1/(1+sw) * g_neg;

    num_A = (k_trail + eff_A) .^ a_trail;
    num_B = (k_trail + eff_B) .^ a_trail;

    P_A = num_A ./ (num_A + num_B);
    est(:,t) = 2 * P_A - 1;
    u(:,t) = 2 * (rand(num_agents,1) < P_A) - 1;

    % pheromone deposition
    q = q_bad * ones(num_agents,1);
    q(u(:,t) == G_t_now) = q_good;

    dep_A = sum(q(u(:,t) == 1));
    dep_B = sum(q(u(:,t) == -1));

    phi_A_ts(t) = (1 - rho) * phi_A_ts(t-1) + dep_A;
    phi_B_ts(t) = (1 - rho) * phi_B_ts(t-1) + dep_B;

    % instantaneous accuracy
    acc(:,t-1) = 1.0 - abs(G_t_now - u(:,t)) / 2.0;
end

end

