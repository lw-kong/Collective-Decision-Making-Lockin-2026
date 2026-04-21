
par_num = 15;
parpool('local',par_num)

rng((now*1000-floor(now*1000))*100000)
filename_save = ['save_Ringv3tx_nu05_k8N40_3_' ...
    datestr(now,30) '_' num2str(randi(99)) '.mat'];

% config
T = 2500;          % time of simulation
dt = 1.0;          % time step
omega_g = 0.1;
noise_sigma = 0.2;
num_agents = 40;  % number of agents

k = 8;
fr_set = 0:0.2:3;
repeat_num = 45;
ring_nu = 0.5;


ring_params = struct('sigma_exc', pi/10, 'factor_decay', 0.1, ...
    'factor_inter', 20, 'factor_input', 0.1, 's_thres', 0, ...
    'num_neuron', 32, 'nu', ring_nu,...
    'gamma', 1, 'dt', 0.1, 'tau', 0.2,...
    'tol', 3e-4, 'Tmax', 2000);

% Run simulations
result_set = zeros(length(fr_set), repeat_num);
tic;
for fr_i = 1:length(fr_set)
    sw = fr_set(fr_i);

    par_temp = zeros(repeat_num,1);
    parfor repeat_i = 1:repeat_num
        rng(repeat_i*20000 + (now*1000-floor(now*1000))*100000)
        network = generate_directed_ER(num_agents,k);
        [acc,~,~,~,~,~] = ...
            func_main_sim_1_net_Ringv3t(sw, num_agents,...
            T, dt, omega_g, noise_sigma, network, ring_params);
        par_temp(repeat_i) = mean(acc(:));
    end
    result_set(fr_i,:) = par_temp;

    save(filename_save)
    toc_now = toc;
    fprintf('%.3f is done\n',fr_i/length(fr_set))
    fprintf('= run time %.2fs\n', toc_now);

end



result_sum = mean(result_set,2);
%[plot_result,best_fr_indx] = max(result_sum,[],2);
save(filename_save)
%% plot
%{

figure('Color','w')
plot(fr_set,result_sum','o-')
hold on
xlabel('fr')
ylabel('global accuracy')
ylim([0.45,1])
title(['WC10d, ER nets, repeat=' num2str(repeat_num)])
grid on
box on

[plot_result,best_fr_indx] = max(result_sum,[],2);
best_fr_set = fr_set(best_fr_indx)';

figure('Color','w')
plot(k_set,plot_result,'o-')
xlabel('k')
ylabel('optimal global accuracy')

figure('Color','w')
plot(k_set,best_fr_set,'o-')
xlabel('k')
ylabel('optimal fr')
%}