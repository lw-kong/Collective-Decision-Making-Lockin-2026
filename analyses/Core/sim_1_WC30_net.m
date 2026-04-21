addpath ..\


% config
num_agents = 128;  % number of agents
k = 8;             % number of neighbors each agent has
T = 2500;          % time of simulation
dt = 1.0;          % time step
omega_g = 0.1;
noise_sigma = 1;

omega_s = 1.5;

%network = generate_ring_network(num_agents);
network = generate_directed_ER(num_agents,k);
message_weight = 1; % =1 DeGroot, =0 naive WC
diz_list = [-1,1]; % =[] DeGroot, =[-1,1] Torney
%diz_list = [-30:0.05:-1,1:0.05:30];
%diz_list = [];

tic
[acc,u,est,m,g,G_ts] = ...
    func_main_sim_1_net_WC30_acc(omega_s, num_agents,...
    T, dt, omega_g, noise_sigma, network, diz_list, message_weight);
toc

%mean(acc(:))

%% plot
load('data_st_color_1.mat')

figure('Position',[600,600,900,480],'Color','w')
plot(G_ts,'--','Color','black','LineWidth',1.5)
hold on
plot(mean(u,1),'Color',c_blue,'LineWidth',1.2)
plot(mean(est,1),'Color',c_green,'LineWidth',1.2)
plot(mean(m,1),'--','Color',c_orange,'LineWidth',1.2)
xlabel('steps')
ylim([-1.2,1.2])
legend('Ground Truth', 'Average Decision','Average Est', 'Average Message')
title(['Torney, ER, N = ' num2str(num_agents) ', k = ' num2str(k)  ...
    ', omega_s = ' num2str(omega_s) ', noise = ' num2str(noise_sigma) ...
    ', acc = ' num2str(mean(acc(:)))])
grid on
box on

%figure('Color','w')
%histogram(sum(network,1))
%xlabel('in-degree')
%title('in-degree distribution')

figure('Color','w')
imagesc(u)
xlabel('steps')
ylabel('agents')
clim([-3,3])
colorbar


%{
figure('Color','w')
histogram(mean(u,1),100)
title(['omega_s = ' num2str(omega_s) ', beta = ' num2str(beta) ', G ratio = ' num2str(G_ratio)])
xlabel('average decision')
ylim([0,2500])
%}

figure('Color','w')
subplot(2,2,1)
imagesc(u)
xlabel('steps')
ylabel('agents')
title('u')
clim([-3,3])
colorbar

subplot(2,2,2)
imagesc(g)
xlabel('steps')
ylabel('agents')
title('g')
clim([-3,3])
colorbar

subplot(2,2,3)
imagesc(est)
xlabel('steps')
ylabel('agents')
title('est')
clim([-3,3])
colorbar

subplot(2,2,4)
imagesc(m)
xlabel('steps')
ylabel('agents')
title('messages')
clim([-3,3])
colorbar
