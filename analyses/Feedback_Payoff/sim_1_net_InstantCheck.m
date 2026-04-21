addpath D:\Research\Misinformation\Model_0

% InstantCheck
% each agent checks own decision correctness with probability check_rate
% if checked and wrong, it flips the social input switch

% config
num_agents = 512;  % number of agents
k = 8;             % number of neighbors each agent has
half_period = 250;
T = 15000;       % time of simulation
dt = 1.0;          % time step
omega_g = 0.1;
noise_sigma = 1;

sw = 4;

diz_list = [-1,1];
mw = 1;

check_rate = 2*1e-4;
%check_rate = 0.1;

fprintf('%.1f checks per period\n',check_rate*num_agents*2*half_period)
fprintf('%.2f%% of all agents checked per period\n',check_rate*2*half_period)

%network = generate_ring_network(num_agents);
network = generate_directed_ER(num_agents,k);

tic
[acc,u,est,m,social_switch,g,G_ts] = ...
    func_main_sim_1_net_InstantCheck_acc(sw, num_agents,...
    T, dt, half_period, omega_g, noise_sigma, network,...
    diz_list, mw, check_rate);
toc

mean(acc(:))

%% plot
load('data_st_color_1.mat')

figure('Position',[600,600,900,480],'Color','w')
plot(G_ts,'--','Color','black','LineWidth',1.5)
hold on
plot(mean(u,1),'Color',c_blue,'LineWidth',1.2)
xlabel('steps')
ylim([-1.2,1.2])
legend('Ground Truth', 'Average Decision')
title(sprintf('instant check, ER, N=%d, k=%d, sw=%.2f, noise=%.2f, check=%.4f, acc=%.3f', ...
    num_agents, k, sw, noise_sigma, check_rate, mean(acc(:))))
grid on
box on

figure('Color','w')
subplot(2,2,1)
imagesc(u)
xlabel('steps')
ylabel('agents')
clim([-1.5,1.5])
colorbar
title('u')

subplot(2,2,2)
imagesc(social_switch)
xlabel('steps')
ylabel('agents')
clim([0,1])
title(sprintf('social switch, instant check, rate=%.4f', check_rate))

subplot(2,2,3)
imagesc(est)
xlabel('steps')
ylabel('agents')
clim([-1.5,1.5])
colorbar
title('est')

subplot(2,2,4)
plot(2*G_ts,'--','Color','black')
hold on
plot(mean(social_switch))
title('ratio of social agents')
ylim([0,1])
grid on


