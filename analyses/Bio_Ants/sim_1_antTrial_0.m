addpath ..\


% config
num_agents = 30;  % number of agents
T = 2500;          % time of simulation
T_half_period = 250;
dt = 1.0;          % time step
omega_g = 0.1;
noise_sigma = 1*10;

sw = 50;

k_trail = 6;      % 经典风格起点
a_trail = 2.0;    % 非线性放大
rho = 0.2;       % 快蒸发
q_good = 1.0;     % new pheromone added if the correct trial is picked
q_bad = 0.2*2;     % new pheromone added if the incorrect trial is picked


tic
[acc,u,est,g,G_ts,phi_A_ts,phi_B_ts] = func_main_sim_1_antTrail_acc_T(...
    sw, num_agents, T, dt, omega_g, noise_sigma, T_half_period, ...
    k_trail, a_trail, rho, q_good, q_bad);
toc

%mean(acc(:))

%% plot
load('data_st_color_1.mat')

figure('Position',[600,300,900,800],'Color','w')
subplot(2,1,1)
plot(G_ts,'--','Color','black','LineWidth',1.5)
hold on
plot(mean(u,1),'Color',c_blue,'LineWidth',1.2)
xlabel('steps')
ylim([-1.2,1.2])
legend('Ground Truth', 'Average Decision')
title(['Torney, ER, N = ' num2str(num_agents)  ...
    ', social weight = ' num2str(sw) ...
    ', acc = ' num2str(mean(acc(:)))])
grid on
box on

subplot(2,1,2)
plot((G_ts+1)/2,'--','Color','black','LineWidth',1.5)
hold on
plot(phi_A_ts/600,'Color',c_blue,'LineWidth',1.2)
plot(phi_B_ts/600,'Color',c_red,'LineWidth',1.2)
ylim([-0.1,1.2])
xlabel('steps')
legend('Ground Truth', 'phi A','phi B')

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
