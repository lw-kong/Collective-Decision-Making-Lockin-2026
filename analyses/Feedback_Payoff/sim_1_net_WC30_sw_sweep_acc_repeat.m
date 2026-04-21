addpath ..\

par_num = 21;   % set to number of cores on your Linux server
parpool('local', par_num)

rng((now*1000-floor(now*1000))*100000)
save_filename = ['save_WC30_sw_0_' datestr(now,30) '_' num2str(randi(999)) '.mat'];




% Sweep sw and evaluate post-transient average accuracy
% For each sw, run repeat_num trials and record one scalar acc per trial:
% mean over agents and time after removing first 10 periods.

% config (aligned with sim_1_net_InstantCheck.m)
num_agents = 512;  % number of agents
k = 8;             % number of neighbors each agent has
half_period = 250;
T = 6000;         % time of simulation
dt = 1.0;          % time step
omega_g = 0.1;
noise_sigma = 1;

diz_list = [-1, 1];
mw = 1;
check_rate = 2e-4;

sw_list = 0:0.1:5;
repeat_num = 500;

period_steps = max(1, round((2 * half_period) / dt));
transient_periods = 2;
transient_steps = transient_periods * period_steps;

acc_trial_all = nan(length(sw_list), repeat_num);

fprintf('sw sweep: %d values, %d repeats each\n', length(sw_list), repeat_num);
fprintf('discard first %d periods (%d steps) as transient\n', transient_periods, transient_steps);

tic
for i_sw = 1:length(sw_list)
    sw = sw_list(i_sw);
    acc_row = nan(1, repeat_num);
    parfor repeat_i = 1:repeat_num
        rng(repeat_i + (now*1000-floor(now*1000))*100000)
        
        network = generate_directed_ER(num_agents, k);

        [acc, ~, ~, ~, ~, ~] = ...
            func_main_sim_1_net_WC30_acc(sw, num_agents, ...
            T, dt, omega_g, noise_sigma, ...
            network, diz_list, mw);

        t_start = transient_steps + 1;
        if t_start > size(acc, 2)
            t_start = 1;
        end

        acc_post = acc(:, t_start:end);
        acc_row(repeat_i) = mean(acc_post(:));
    end
    acc_trial_all(i_sw, :) = acc_row;

    fprintf('sw = %.1f done (%d/%d), mean acc = %.4f\n', ...
        sw, i_sw, length(sw_list), mean(acc_trial_all(i_sw, :), 'omitnan'));
    toc
end
toc

acc_mean_vs_sw = mean(acc_trial_all, 2, 'omitnan');
acc_std_vs_sw = std(acc_trial_all, 0, 2, 'omitnan');


time_consumed = toc;
save(save_filename)
fprintf('Workspace 已保存至 %s\n', save_filename);

%%
%{
figure('Color', 'w')
errorbar(sw_list, acc_mean_vs_sw, acc_std_vs_sw, '-o', ...
    'LineWidth', 1.2, 'MarkerSize', 4)
xlabel('sw')
ylabel('post-transient mean acc')
title(sprintf('Accuracy vs sw (%d repeats per sw)', repeat_num))
grid on
%}
