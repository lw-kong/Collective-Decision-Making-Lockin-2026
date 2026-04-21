addpath ..\

% ring attractor type decision-making
% the two env states: +pi/2 or -pi/2


% config
num_agents = 40;%32;  % number of agents
k = 4*2;             % number of neighbors each agent has
T = 2500;          % time of simulation
dt = 1.0;          % time step
omega_g = 0.1;
noise_sigma = 0.2;

social_weight = 5;

%network = generate_ring_network(num_agents);
network = generate_directed_ER(num_agents,k);

ring_params = struct('sigma_exc', pi/10, 'factor_decay', 0.1, ...
    'factor_inter', 20, 'factor_input', 0.1, 's_thres', 0, ...
    'num_neuron', 64, 'nu', 0.5,...
    'gamma', 1, 'dt', 0.1, 'tau', 0.2,...
    'tol', 5e-4, 'Tmax', 2000);

ring_params = struct('sigma_exc', pi/10, 'factor_decay', 0.1, ...
    'factor_inter', 20, 'factor_input', 0.1, 's_thres', 0, ...
    'num_neuron', 32, 'nu', 0.75,...
    'gamma', 1, 'dt', 0.1, 'tau', 0.2,...
    'tol', 5e-4, 'Tmax', 2000);

tic
[acc,u,u_theta,g_theta,g,G_ts,iter_all] = ...
    func_main_sim_1_net_Ringv3ty(social_weight, num_agents,...
    T, dt, omega_g, noise_sigma, network, ring_params);
toc


%% plot
load('data_st_color_1.mat')

%
figure('Position',[600,600,900,480],'Color','w')
plot(G_ts,'--','Color','black','LineWidth',1.5)
hold on
plot(mean(u,1),'Color',c_blue,'LineWidth',1.2)
xlabel('steps')
ylim([-1.2,1.2])
legend('Ground Truth', 'Average Decision')
title(['Ring0, 1D, N = ' num2str(num_agents) ', k = ' num2str(k)  ...
    ', nu = ' num2str(ring_params.nu) ...
    ', neurons = ' num2str(ring_params.num_neuron) ...
    ', sw = ' num2str(social_weight) ', noise = ' num2str(noise_sigma) ...
    ', acc = ' num2str(mean(acc(:)))])
grid on
box on
%


figure('Color','w','Position',[1200,400,1000,400])
subplot(2,2,1)
imagesc(u)
xlabel('steps')
ylabel('agents')
clim([-1,1])
colorbar
title('u')

subplot(2,2,3)
imagesc(wrapToPi(u_theta) /pi)
xlabel('steps')
ylabel('agents')
clim([-1,1])
colorbar
title('u theta / pi')

subplot(2,2,2)
imagesc(wrapToPi(g_theta) /pi)
xlabel('steps')
ylabel('agents')
clim([-1,1])
colorbar
title('g theta /pi')

subplot(2,2,4)
imagesc(g)
xlabel('steps')
ylabel('agents')
clim([-3,3])
colorbar
title('g')


plot_ind = [300:480,800:980];
figure('Color','w')
histogram(wrapToPi(g_theta(:,plot_ind)) / pi*2,'Normalization','pdf')
xlabel('g theta / pi/2')
ylabel('probability density')
%ylim([0,1])
xlim([-2,2])
title(['distribution of g theta / pi in the G=1 region, noise = ' ...
    num2str(noise_sigma)])

figure()
histogram(iter_all(:))
xlabel('iterations run')

%{
plot_ind = [3200:3800];
figure('Color','w')
histogram(wrapToPi(u_theta(:,plot_ind)) / pi,'Normalization','pdf')
xlabel('u theta / pi')
ylabel('probability density')
%ylim([0,1])
title(['distribution of u theta / pi in the G=-1 region, noise = ' ...
    num2str(noise_sigma)])
%}

