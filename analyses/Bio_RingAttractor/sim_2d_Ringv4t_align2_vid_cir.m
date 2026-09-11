
% 2-D ring-attractor collective decision-making simulation.
% Agents move in the plane, interact with neighbors inside a metric radius,
% and integrate noisy resource direction with social heading cues.

%% config
rng(1) % for sw = 0.5, 1
%rng(5) % for sw = 1.2
%rng('shuffle')

num_agents = 40;
T = 200;                  % number of simulation steps
dt = 1.0;
step_length = 0.2;        % distance moved by each agent per step

social_weight = 1;

interaction_radius = 8.0;
observation_weight = 1.0;
obs_noise_sigma = 0.35 + 0.65;   % angular noise in radians

arena_radius = 20;
resource_distance = 120;  % endpoints are far enough to define broad directions
%resource_angles = [pi/4, 5*pi/4];
resource_angles = [pi/4, 3*pi/4];
%resource_angles = [0, 2*pi/3];
resource_half_period = 100;
env_mode = 'cir';  % 'switch', 'linear', 'pos', or 'cir'
resource_motion = env_mode;

% Used when env_mode = 'pos'. Each row is [start_step, end_step, x, y].
resource_schedule = [
    1,   100,  resource_distance*cos(pi/4),   resource_distance*sin(pi/4);
    101, 200,  resource_distance*cos(3*pi/4), resource_distance*sin(3*pi/4)
];

% Used when env_mode = 'cir'.
resource_circle_radius = 120;
resource_circle_period = 200;
resource_circle_phase0 = pi/4;

make_video = true;
video_filename = ['v2_Ringv4t_align2_', env_mode,...
    '_sw', num2str(social_weight) ...
    , '_', datestr(now,30), '.mp4'];
video_fps = 20;
video_stride = 1;
video_colormap = parula(256);

ring_params = struct('sigma_exc', pi/10, 'factor_decay', 0.1, ...
    'factor_inter', 20, 'factor_input', 0.1, 's_thres', 0, ...
    'num_neuron', 32, 'nu', 0.5, ...
    'gamma', 1, 'dt', 0.1, 'tau', 0.2, ...
    'tol', 3e-4, 'Tmax', 2000);

sim_params = struct();
sim_params.num_agents = num_agents;
sim_params.T = T;
sim_params.dt = dt;
sim_params.step_length = step_length;
sim_params.interaction_radius = interaction_radius;
sim_params.social_weight = social_weight;
sim_params.observation_weight = observation_weight;
sim_params.obs_noise_sigma = obs_noise_sigma;
sim_params.arena_radius = arena_radius;
sim_params.resource_distance = resource_distance;
sim_params.resource_angles = resource_angles;
sim_params.resource_half_period = resource_half_period;
sim_params.resource_motion = resource_motion;
sim_params.env_mode = env_mode;
sim_params.resource_schedule = resource_schedule;
sim_params.resource_circle_radius = resource_circle_radius;
sim_params.resource_circle_period = resource_circle_period;
sim_params.resource_circle_phase0 = resource_circle_phase0;

%% run simulation
tic
result = func_main_sim_2d_Ringv4t_align2(sim_params, ring_params);
toc

%% summary plots
figure('Color','w','Position',[200,200,1400,420])
subplot(1,3,1)
plot(result.positions(:,1,1), result.positions(:,2,1), 'o', ...
    'Color', [0.6,0.6,0.6])
hold on
plot(result.positions(:,1,end), result.positions(:,2,end), 'o', ...
    'MarkerFaceColor', [0.1,0.35,0.9], 'MarkerEdgeColor', 'none')
plot(result.resource_plot_pos(:,1), result.resource_plot_pos(:,2), ...
    'r.', 'MarkerSize', 8)
axis equal
grid on
box on
xlabel('x')
ylabel('y')
legend('initial agents', 'final agents', 'resource', 'Location', 'best')
title('2-D trajectories summary')

subplot(1,3,2)
plot(result.mean_alignment_to_resource, 'LineWidth', 1.3)
hold on
ylim([-1,1])
grid on
box on
xlabel('steps')
title('collective tracking (mean cos heading-resource)')

subplot(1,3,3)
plot(result.mean_neighbor_count, 'LineWidth', 1.3)
title('mean neighbor count')
grid on
box on
xlabel('steps')

%% video
if make_video
    func_make_video_2d_Ringv4t_align2(result, sim_params, video_filename, ...
        video_fps, video_stride, video_colormap);
    fprintf('Saved video: %s\n', video_filename);
end
