addpath ..\

% Sweep social-weight parameter sw
% Output curves:
%   1) average accuracy vs sw
%   2) average max(phi_A_ts) vs sw



% ------------------ base config (copied from single-run script) ------------------
num_agents = 50;      % number of agents
T = 2500;              % total simulation time
T_half_period = 250;   % environment flip half-period
dt = 1.0;              % time step
omega_g = 0.1;
noise_sigma = 1;

k_trail = 6;
a_trail = 2.0;
rho = 0.5;
q_good = 1.0*0.4;
q_bad = 0.8*0.4;

% ------------------ sweep setup ------------------
%sw_list = 0:0.1:1.5;  % adjust range/density as needed
sw_list = 0:0.2:10;
%sw_list = 0:2:50;
%sw_list = 0:0.25:6;
num_repeat = 500;                 % repeated runs per sw for smoother statistics

avg_acc_vs_sw = zeros(size(sw_list));
avg_max_phiA_vs_sw = zeros(size(sw_list));

%rng(1); % reproducible sweep
%%
tic;
for para_i = 1:numel(sw_list)
    sw = sw_list(para_i);

    acc_runs = zeros(num_repeat, 1);
    max_phiA_runs = zeros(num_repeat, 1);

    for r = 1:num_repeat
        [acc, ~, ~, ~, ~, phi_A_ts, ~] = func_main_sim_1_antTrail_acc_T( ...
            sw, num_agents, T, dt, omega_g, noise_sigma, T_half_period, ...
            k_trail, a_trail, rho, q_good, q_bad);

        acc_runs(r) = mean(acc(:));
        max_phiA_runs(r) = max(phi_A_ts);
    end

    avg_acc_vs_sw(para_i) = mean(acc_runs);
    avg_max_phiA_vs_sw(para_i) = mean(max_phiA_runs);

    fprintf('sw = %.3f | mean(acc) = %.4f | mean(max(phi_A)) = %.4f\n', ...
        sw, avg_acc_vs_sw(para_i), avg_max_phiA_vs_sw(para_i));
end
toc;

%% ------------------ plot ------------------
figure('Position', [600, 260, 920, 760], 'Color', 'w');

subplot(2,1,1);
plot(sw_list, avg_acc_vs_sw, '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
xlabel('sw');
ylabel('Average acc');
title(['Average Accuracy vs sw | N = ' num2str(num_agents) ...
    ', \rho = ' num2str(rho) ...
    ', k_{trail} = ' num2str(k_trail) ...
    ', a_{trail} = ' num2str(a_trail) ', q_{bad} = ' num2str(q_bad)]);
grid on;
box on;

subplot(2,1,2);
plot(sw_list, avg_max_phiA_vs_sw, '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
xlabel('sw');
ylabel('Average max(phi\_A\_ts)');
title('Average max(phi\_A\_ts) vs sw');
grid on;
box on;

