
% Sweep social_weight under the align2 fixed-position resource environment.
% No video is generated. Each social_weight is repeated repeat_num times.
% Summary statistics are computed only over steps 101:200.

%% config
social_weight_set = 0:0.1:3.5;
repeat_num = 10;
base_seed = round((now*1000-floor(now*1000))*100000);

use_parallel = true;
par_num = 5;

num_agents = 40;
T = 200;                  % number of simulation steps
dt = 1.0;
step_length = 0.1;        % distance moved by each agent per step
analysis_steps = 101:200;

interaction_radius = 8.0;
observation_weight = 1.0;
obs_noise_sigma = 1.0;   % angular noise in radians

arena_radius = 20;
resource_distance = 120;  % endpoints are far enough to define broad directions
resource_angles = [pi/4, 3*pi/4];
resource_half_period = 100;
env_mode = 'pos';
resource_motion = env_mode;

% Used when env_mode = 'pos'. Each row is [start_step, end_step, x, y].
resource_schedule = [
    1,   100,  resource_distance*cos(pi/4),   resource_distance*sin(pi/4);
    101, 200,  resource_distance*cos(3*pi/4), resource_distance*sin(3*pi/4)
];

% Circular parameters are kept for compatibility with align2 sim_params.
resource_circle_radius = 120;
resource_circle_period = 200;
resource_circle_phase0 = pi/4;

save_results = true;
filename_save = ['save_sweep_social_weight_2d_Ringv4t_align2_pos_', ...
    datestr(now,30), '.mat'];

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

if use_parallel
    pool_now = gcp('nocreate');
    if isempty(pool_now)
        parpool('local', par_num);
    end
end

%% sweep social weight
len_time_steps = round(T / dt);
if max(analysis_steps) > len_time_steps
    error('analysis_steps exceeds the simulation length.');
end

num_social_weights = numel(social_weight_set);
alignment_mean_set = zeros(num_social_weights, 1);
neighbor_count_mean_set = zeros(num_social_weights, 1);
alignment_sem_set = zeros(num_social_weights, 1);
neighbor_count_sem_set = zeros(num_social_weights, 1);

alignment_trial_mean_all = zeros(num_social_weights, repeat_num);
neighbor_count_trial_mean_all = zeros(num_social_weights, repeat_num);
alignment_curve_mean_all = zeros(num_social_weights, len_time_steps);
neighbor_count_curve_mean_all = zeros(num_social_weights, len_time_steps);
accepted_seed_set = zeros(num_social_weights, repeat_num);

tic
for sw_i = 1:num_social_weights
    social_weight = social_weight_set(sw_i);
    sim_params_sw = sim_params;
    sim_params_sw.social_weight = social_weight;

    alignment_curve_set = zeros(repeat_num, len_time_steps);
    neighbor_count_curve_set = zeros(repeat_num, len_time_steps);
    alignment_trial_mean_set = zeros(repeat_num, 1);
    neighbor_count_trial_mean_set = zeros(repeat_num, 1);
    accepted_seed_row = zeros(1, repeat_num);

    parfor repeat_i = 1:repeat_num
        seed_now = base_seed + (sw_i - 1) * repeat_num + repeat_i - 1;
        rng(seed_now)

        result = func_main_sim_2d_Ringv4t_align2(sim_params_sw, ring_params);
        alignment_curve_set(repeat_i,:) = result.mean_alignment_to_resource;
        neighbor_count_curve_set(repeat_i,:) = result.mean_neighbor_count;
        alignment_trial_mean_set(repeat_i) = mean(...
            result.mean_alignment_to_resource(analysis_steps));
        neighbor_count_trial_mean_set(repeat_i) = mean(...
            result.mean_neighbor_count(analysis_steps));
        accepted_seed_row(repeat_i) = seed_now;

        %fprintf('sw %.3f repeat %d/%d is done\n', ...
        %    social_weight, repeat_i, repeat_num);
    end

    alignment_trial_mean_all(sw_i,:) = alignment_trial_mean_set;
    neighbor_count_trial_mean_all(sw_i,:) = neighbor_count_trial_mean_set;
    alignment_curve_mean_all(sw_i,:) = mean(alignment_curve_set, 1);
    neighbor_count_curve_mean_all(sw_i,:) = mean(neighbor_count_curve_set, 1);
    accepted_seed_set(sw_i,:) = accepted_seed_row;

    alignment_mean_set(sw_i) = mean(alignment_trial_mean_set);
    neighbor_count_mean_set(sw_i) = mean(neighbor_count_trial_mean_set);
    alignment_sem_set(sw_i) = std(alignment_trial_mean_set) / sqrt(repeat_num);
    neighbor_count_sem_set(sw_i) = std(neighbor_count_trial_mean_set) / ...
        sqrt(repeat_num);

    fprintf(['social_weight %.3f done: average alignment = %.4f, ' ...
        'average neighbors = %.4f\n'], social_weight, ...
        alignment_mean_set(sw_i), neighbor_count_mean_set(sw_i));

    if save_results
        save(filename_save)
    end
end
toc

if save_results
    save(filename_save)
    fprintf('Saved results: %s\n', filename_save);
end

%% plot sweep summary
figure('Color','w','Position',[200,200,1200,420])
subplot(1,2,1)
errorbar(social_weight_set, alignment_mean_set, alignment_sem_set, ...
    'o-', 'Color', [0.1,0.35,0.9], 'MarkerFaceColor', [0.1,0.35,0.9], ...
    'LineWidth', 1.5)
ylim([-1,1])
grid on
box on
xlabel('social weight')
ylabel('average alignment to resource')
title(['pos resource, steps ' num2str(analysis_steps(1)) '-' ...
    num2str(analysis_steps(end))])

subplot(1,2,2)
errorbar(social_weight_set, neighbor_count_mean_set, ...
    neighbor_count_sem_set, 'o-', 'Color', [0.85,0.2,0.1], ...
    'MarkerFaceColor', [0.85,0.2,0.1], 'LineWidth', 1.5)
grid on
box on
xlabel('social weight')
ylabel('average neighbor count')
title(['repeat = ' num2str(repeat_num) ', analysis steps = 101-200'])

%% optional time-course overview
figure('Color','w','Position',[260,260,1200,420])
t_axis = 1:len_time_steps;
subplot(1,2,1)
imagesc(t_axis, social_weight_set, alignment_curve_mean_all)
axis xy
colorbar
clim([-1,1])
xlabel('steps')
ylabel('social weight')
title('mean alignment time course')

subplot(1,2,2)
imagesc(t_axis, social_weight_set, neighbor_count_curve_mean_all)
axis xy
colorbar
xlabel('steps')
ylabel('social weight')
title('mean neighbor count time course')
