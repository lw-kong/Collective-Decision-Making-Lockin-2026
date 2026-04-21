
% TopKM
% only top k among the input messages. o is always received

addpath 'D:\Research\Misinformation\Model_0\gen_3_Matlab'
%parpool('local',6)

rng((now*1000-floor(now*1000))*100000)
%filename_save = ['save_WC30_N128_repeat20_' ...
%    datestr(now,30) '_' num2str(randi(99)) '.mat'];

% config
num_agents = 512;  % number of agents
k = 8;             % number of neighbors each agent has
T = 2500;          % time of simulation
T_half_period = 250;
dt = 1.0;          % time step
omega_g = 0.1;
noise_sigma = 1;

diz_list = [];
mw = 1;  % =1 DeGroot, =0 naive WC

%para_set = [1:0.05:2,2.1:0.1:6,6:0.2:8];
para_set = 0:0.025:2;
repeat_num = 250;            % number of repeats

top_k = 2;

% Run simulations
result_set = zeros(length(para_set), repeat_num);
tic;
for para_i = 1:length(para_set)
    sw = para_set(para_i);
    
    for repeat_i = 1:repeat_num
        network = generate_directed_ER(num_agents,k);
        [acc,~,~,~,~,~] = ...
            func_main_sim_1_net_WC30_TopKm(sw, num_agents,...
            T, dt, omega_g, noise_sigma, ...
            network, diz_list,  mw, top_k);
        result_set(para_i, repeat_i) = mean(acc(:));
    end
    
    toc_now = toc;
    %save(filename_save)
    %if mod(para_i,4) == 0
        fprintf('sw = %.3f\n', sw);
        toc_now = toc;
        fprintf('= run time %.2fs, %.2f is done\n', ...
            toc_now, para_i/length(para_set));
    %end
end

%% plot
%
figure('Color','w')
plot(para_set,mean(result_set,2),'o-')
xlabel('omega s')
ylabel('global accuracy')
ylim([0.45,1])
title(['Top-K, ER, Top ' num2str(top_k) ', N = ' num2str(num_agents) ...
    ', k = ' num2str(k) ', repeat = ' num2str(repeat_num)])
grid on
box on
%