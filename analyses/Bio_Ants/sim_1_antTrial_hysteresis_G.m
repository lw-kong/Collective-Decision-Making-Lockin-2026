addpath ..\

% Hysteresis under slow, noise-free environment sweep
% G goes low->high (up branch) and high->low (down branch)

% ------------------ model config ------------------
num_agents = 100;
sw = 5;
k_trail = 6;
a_trail = 2.0;
rho = 0.2;
q_good = 1.0;
q_bad = 0.2;


num_agents = 50;
sw = 3;
k_trail = 6;
a_trail = 2.0;
rho = 0.5;
q_good = 1.0*0.4;
q_bad = 0.8*0.4;


% ------------------ hysteresis sweep config ------------------
G_min = -3;
G_max = 3;
n_steps_branch = 500*20;      % larger means slower sweep
num_repeat = 20;            % draw every run, no averaging

G_up = linspace(G_min, G_max, n_steps_branch);
G_down = linspace(G_max, G_min, n_steps_branch);

mean_u_up_runs = zeros(num_repeat, n_steps_branch);
mean_u_down_runs = zeros(num_repeat, n_steps_branch);
mean_phiA_up_runs = zeros(num_repeat, n_steps_branch);
mean_phiA_down_runs = zeros(num_repeat, n_steps_branch);

%rng(1); % reproducible

tic;
for r = 1:num_repeat
    % no-noise external private cue: every agent sees the same scalar G(t)
    g_up = repmat(G_up, num_agents, 1);
    g_down = repmat(G_down, num_agents, 1);

    [~, u_up, ~, ~, ~, phi_A_up, ~] = func_main_sim_1_antTrail_acc_T_g( ...
        sw, g_up, k_trail, a_trail, rho, q_good, q_bad);
    [~, u_down, ~, ~, ~, phi_A_down, ~] = func_main_sim_1_antTrail_acc_T_g( ...
        sw, g_down, k_trail, a_trail, rho, q_good, q_bad);

    mean_u_up_runs(r, :) = mean(u_up, 1);
    mean_u_down_runs(r, :) = mean(u_down, 1);
    mean_phiA_up_runs(r, :) = phi_A_up';
    mean_phiA_down_runs(r, :) = phi_A_down';
end
toc;

% ------------------ plot hysteresis loop ------------------
figure('Position', [620, 260, 860, 760], 'Color', 'w');

subplot(2,1,1);
hold on;
for r = 1:num_repeat
    plot(G_up, mean_u_up_runs(r, :), '-', 'LineWidth', 1.0, 'Color', [0 0.4470 0.7410]);
    plot(G_down, mean_u_down_runs(r, :), '-', 'LineWidth', 1.0, 'Color', [0.8500 0.3250 0.0980]);
end
xlabel('G (external cue)');
ylabel('<u> (collective decision)');
title(sprintf('Hysteresis of collective choice (sw=%.2f, N=%d, k_{trail}=%.1f, a_{trail}=%.1f, \\rho=%.2f)', ...
    sw, num_agents, k_trail, a_trail, rho));
%legend('Up sweep: G_{min} \rightarrow G_{max}', ...
%       'Down sweep: G_{max} \rightarrow G_{min}', ...
%       'Location', 'best');
grid on;
box on;
ylim([-1.05, 1.05]);

subplot(2,1,2);
hold on;
for r = 1:num_repeat
    plot(G_up, mean_phiA_up_runs(r, :), '-', 'LineWidth', 1.0, 'Color', [0 0.4470 0.7410]);
    plot(G_down, mean_phiA_down_runs(r, :), '-', 'LineWidth', 1.0, 'Color', [0.8500 0.3250 0.0980]);
end
xlabel('G (external cue)');
ylabel('<\phi_A>');
title('Hysteresis of pheromone on path A');
legend('Up sweep: G_{min} \rightarrow G_{max}', ...
       'Down sweep: G_{max} \rightarrow G_{min}', ...
       'Location', 'best');
grid on;
box on;

