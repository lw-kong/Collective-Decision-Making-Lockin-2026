function result = func_main_sim_2d_Ringv4t_align2(sim_params, ring_params)
% Run the align2 2-D collective decision-making model.
% Environment modes:
%   switch/linear: legacy two-resource endpoint dynamics.
%   pos: static resources from [start_step, end_step, x, y] rows.
%   cir: one resource rotating around the origin.

num_agents = sim_params.num_agents;
len_time_steps = round(sim_params.T / sim_params.dt);

positions = zeros(num_agents, 2, len_time_steps);
heading_theta = zeros(num_agents, len_time_steps);
obs_theta = nan(num_agents, len_time_steps);
resource_theta = nan(num_agents, len_time_steps);
neighbor_count = zeros(num_agents, len_time_steps);
alignment_to_resource = nan(num_agents, len_time_steps);

resource_pos_cell = cell(len_time_steps, 1);
resource_plot_pos = nan(len_time_steps, 2);
resource_state = zeros(len_time_steps, 1);
active_resource_count = zeros(len_time_steps, 1);

% Initial positions are spread inside a disk near the origin.
init_r = sim_params.arena_radius * sqrt(rand(num_agents, 1));
init_phi = 2*pi*rand(num_agents, 1);
positions(:,:,1) = [init_r .* cos(init_phi), init_r .* sin(init_phi)];
heading_theta(:,1) = 2*pi*rand(num_agents, 1);

for t = 1:len_time_steps
    [resource_pos_cell{t}, resource_state(t)] = get_resource_positions(...
        t, sim_params);
    active_resource_count(t) = size(resource_pos_cell{t}, 1);
    if active_resource_count(t) > 0
        resource_plot_pos(t,:) = mean(resource_pos_cell{t}, 1);
    end
end

for t = 1:len_time_steps
    current_positions = positions(:,:,t);
    current_resources = resource_pos_cell{t};
    D = pairwise_distances(current_positions);
    adjacency = (D <= sim_params.interaction_radius) & (D > 0);
    neighbor_count(:,t) = sum(adjacency, 2);

    for i = 1:num_agents
        true_resource_theta = angles_to_targets(current_positions(i,:), ...
            current_resources);
        if ~isempty(true_resource_theta)
            resource_theta(i,t) = circular_mean(true_resource_theta);
            obs_theta(i,t) = wrapTo2Pi(resource_theta(i,t) + ...
                sim_params.obs_noise_sigma * randn());
            alignment_to_resource(i,t) = cos(wrapToPi(heading_theta(i,t) - ...
                resource_theta(i,t)));
        end
    end

    if t == len_time_steps
        break
    end

    for i = 1:num_agents
        selected_indices = adjacency(i,:);

        input_ob = observation_inputs(current_positions(i,:), ...
            current_resources, sim_params);
        input_social = [heading_theta(selected_indices,t), ...
            sim_params.social_weight * ones(sum(selected_indices), 1)];
        input0 = [input_ob; input_social];

        if isempty(input0)
            heading_theta(i,t+1) = heading_theta(i,t);
        else
            heading_theta(i,t+1) = func_ring_v4t(input0, ring_params);
            if isnan(heading_theta(i,t+1))
                heading_theta(i,t+1) = heading_theta(i,t);
            end
        end
    end

    positions(:,1,t+1) = positions(:,1,t) + ...
        sim_params.step_length * cos(heading_theta(:,t+1));
    positions(:,2,t+1) = positions(:,2,t) + ...
        sim_params.step_length * sin(heading_theta(:,t+1));
end

result = struct();
result.positions = positions;
result.heading_theta = heading_theta;
result.obs_theta = obs_theta;
result.resource_theta = resource_theta;
result.resource_pos = resource_plot_pos;
result.resource_plot_pos = resource_plot_pos;
result.resource_pos_cell = resource_pos_cell;
result.resource_state = resource_state;
result.active_resource_count = active_resource_count;
result.neighbor_count = neighbor_count;
result.alignment_to_resource = alignment_to_resource;
result.mean_alignment_to_resource = mean_finite_rows(alignment_to_resource);
result.mean_neighbor_count = mean(neighbor_count, 1);
end

function input_ob = observation_inputs(agent_xy, resource_xy, sim_params)
theta = angles_to_targets(agent_xy, resource_xy);
if isempty(theta)
    input_ob = zeros(0, 2);
    return
end
theta = wrapTo2Pi(theta + sim_params.obs_noise_sigma * randn(size(theta)));
input_ob = [theta, sim_params.observation_weight * ones(numel(theta), 1)];
end

function [resource_xy, state] = get_resource_positions(t, sim_params)
if isfield(sim_params, 'env_mode')
    env_mode = sim_params.env_mode;
elseif isfield(sim_params, 'resource_motion')
    env_mode = sim_params.resource_motion;
else
    env_mode = 'linear';
end

switch lower(env_mode)
    case 'pos'
        schedule = sim_params.resource_schedule;
        is_active = (t >= schedule(:,1)) & (t <= schedule(:,2));
        resource_xy = schedule(is_active, 3:4);
        state = sum(is_active);

    case 'cir'
        theta = sim_params.resource_circle_phase0 + ...
            2*pi*(t-1)*sim_params.dt / sim_params.resource_circle_period;
        resource_xy = sim_params.resource_circle_radius * ...
            [cos(theta), sin(theta)];
        state = wrapTo2Pi(theta);

    case {'switch', 'linear'}
        [resource_xy, state] = endpoint_resource_position(t, sim_params, ...
            lower(env_mode));

    otherwise
        error(['Unknown env_mode/resource_motion: %s. Use ''switch'', ' ...
            '''linear'', ''pos'', or ''cir''.'], env_mode);
end
end

function [resource_xy, state] = endpoint_resource_position(t, sim_params, env_mode)
endpoint_theta = sim_params.resource_angles(:);
endpoints = sim_params.resource_distance * ...
    [cos(endpoint_theta), sin(endpoint_theta)];
one_way_steps = sim_params.resource_half_period;

switch env_mode
    case 'switch'
        phase = floor((t-1) / one_way_steps);
        state = mod(phase, 2) + 1;
        resource_xy = endpoints(state,:);

    case 'linear'
        phase_time = mod(t-1, 2*one_way_steps);
        if phase_time <= one_way_steps
            state = phase_time / one_way_steps;
        else
            state = 2 - phase_time / one_way_steps;
        end
        resource_xy = (1 - state) * endpoints(1,:) + state * endpoints(2,:);
end
end

function theta = angles_to_targets(agent_xy, target_xy)
if isempty(target_xy)
    theta = zeros(0, 1);
    return
end
delta = target_xy - agent_xy;
theta = atan2(delta(:,2), delta(:,1));
theta = wrapTo2Pi(theta);
end

function theta = circular_mean(theta_set)
theta = atan2(sum(sin(theta_set)), sum(cos(theta_set)));
theta = wrapTo2Pi(theta);
end

function D = pairwise_distances(X)
dx = X(:,1) - X(:,1).';
dy = X(:,2) - X(:,2).';
D = sqrt(dx.^2 + dy.^2);
end

function y = mean_finite_rows(X)
y = nan(1, size(X, 2));
for col_i = 1:size(X, 2)
    x = X(:, col_i);
    x = x(isfinite(x));
    if ~isempty(x)
        y(col_i) = mean(x);
    end
end
end

function theta = wrapTo2Pi(theta)
theta = mod(theta, 2*pi);
end

function theta = wrapToPi(theta)
theta = mod(theta + pi, 2*pi) - pi;
end
