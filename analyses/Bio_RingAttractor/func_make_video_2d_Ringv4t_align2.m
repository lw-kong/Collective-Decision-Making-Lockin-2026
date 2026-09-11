function func_make_video_2d_Ringv4t_align2(result, sim_params, filename, ...
    fps, stride, video_colormap)
% Create an MP4 movie for the align2 2-D simulation.
% Left: agents + target motion. Right: collective tracking curve with
% a marker that follows the current time step.

if nargin < 6 || isempty(video_colormap)
    video_colormap = parula(256);
end
if nargin < 5
    stride = 1;
end
if nargin < 4
    fps = 20;
end

positions = result.positions;
heading_theta = result.heading_theta;
neighbor_count = result.neighbor_count;
resource_pos_cell = result.resource_pos_cell;
mean_alignment = result.mean_alignment_to_resource(:);

num_steps = size(positions, 3);
all_x = reshape(positions(:,1,:), [], 1);
all_y = reshape(positions(:,2,:), [], 1);
resource_all = vertcat_nonempty(resource_pos_cell);
if ~isempty(resource_all)
    all_x = [all_x; resource_all(:,1)];
    all_y = [all_y; resource_all(:,2)];
end
pad = 0.08 * max(max(all_x) - min(all_x), max(all_y) - min(all_y));
if pad == 0
    pad = 1;
end
x_lim = [min(all_x)-pad, max(all_x)+pad];
y_lim = [min(all_y)-pad, max(all_y)+pad];

max_neighbor_count = max(neighbor_count(:));
if max_neighbor_count == 0
    max_neighbor_count = 1;
end

steps = (1:numel(mean_alignment))';
line_color = [0.20, 0.45, 0.78];
marker_color = [0.90, 0.35, 0.18];

if isfield(sim_params, 'env_mode')
    env_mode = sim_params.env_mode;
else
    env_mode = sim_params.resource_motion;
end

switch lower(env_mode)
    case 'pos'
        env_name = 'Target Switch';
    case 'cir'
        env_name = 'Moving Target';
    otherwise
        env_name = env_mode;
end

writer = VideoWriter(filename, 'MPEG-4');
writer.FrameRate = fps;
open(writer);

% Wide canvas: left spatial panel dominates, right curve is a side strip.
fig = figure('Color','w','Position',[60,60,1300,650], 'Visible','on');

for t = 1:stride:num_steps
    clf(fig)

    % --- left: spatial scene (larger) ---
    ax = axes('Parent', fig, 'Position', [0.055, 0.10, 0.52, 0.80]);
    hold(ax, 'on')

    pos_t = positions(:,:,t);
    heading_t = heading_theta(:,t);
    resources_t = resource_pos_cell{t};

    scatter(ax, pos_t(:,1), pos_t(:,2), 64, neighbor_count(:,t), ...
        'filled', 'MarkerEdgeColor', 'none');
    colormap(ax, video_colormap)
    caxis(ax, [0, max_neighbor_count])
    cb = colorbar(ax);
    cb.Label.String = 'number of neighbors inside radius';
    cb.Label.FontSize = 11;

    quiver(ax, pos_t(:,1), pos_t(:,2), cos(heading_t), sin(heading_t), ...
        0.55, 'Color', [0.18,0.18,0.18], 'LineWidth', 1.05, ...
        'MaxHeadSize', 1.15);

    if ~isempty(resources_t)
        plot(ax, resources_t(:,1), resources_t(:,2), 'p', ...
            'MarkerFaceColor', [0.92,0.18,0.18], ...
            'MarkerEdgeColor', 'none', 'MarkerSize', 18, ...
            'LineStyle', 'none');
    end

    if isfield(result, 'resource_plot_pos')
        plot(ax, result.resource_plot_pos(:,1), result.resource_plot_pos(:,2), ...
            '--', 'Color', [0.88,0.35,0.30], 'LineWidth', 1.15);
    end

    axis(ax, 'equal')
    xlim(ax, x_lim)
    ylim(ax, y_lim)
    grid(ax, 'on')
    box(ax, 'on')
    ax.GridAlpha = 0.14;
    ax.LineWidth = 0.9;
    ax.FontSize = 13;
    xlabel(ax, '$x$',...
    'Interpreter', 'latex', 'FontSize', 17)
    ylabel(ax, '$y$',...
    'Interpreter', 'latex', 'FontSize', 17)
    title(ax, sprintf(['Collective Tracking with Ring Attractor Networks | %s \n'...
        'Time Step %d/%d | ' ...
        'Social Weight = %.2f'], ...
        env_name, t, num_steps, ...
        sim_params.social_weight), 'FontSize', 15, 'FontWeight', 'normal')

    % --- right: collective tracking curve + time marker ---
    ax2 = axes('Parent', fig, 'Position', [0.66, 0.18, 0.30, 0.64]);
    hold(ax2, 'on')

    % Zero reference
    plot(ax2, [1, num_steps], [0, 0], '-', ...
        'Color', [0.82,0.82,0.82], 'LineWidth', 1.0)

    % Full curve (soft) + elapsed portion (stronger)
    plot(ax2, steps, mean_alignment, '-', ...
        'Color', [0.72, 0.82, 0.93], 'LineWidth', 1.5)
    plot(ax2, steps(1:t), mean_alignment(1:t), '-', ...
        'Color', line_color, 'LineWidth', 2.2)

    % Current-time vertical guide
    plot(ax2, [t, t], [-1, 1], '-', ...
        'Color', [0.78,0.78,0.78], 'LineWidth', 1.0)

    % Current-state marker
    plot(ax2, t, mean_alignment(t), 'o', ...
        'MarkerSize', 11, ...
        'MarkerFaceColor', marker_color, ...
        'MarkerEdgeColor', 'w', ...
        'LineWidth', 1.6)

    ylim(ax2, [-1, 1])
    xlim(ax2, [1, num_steps])
    grid(ax2, 'on')
    box(ax2, 'off')
    ax2.GridAlpha = 0.18;
    ax2.GridColor = [0.55,0.55,0.55];
    ax2.LineWidth = 0.9;
    ax2.FontSize = 13;
    ax2.TickDir = 'out';
    ax2.YAxisLocation = 'left';
    xlabel(ax2, 'Time Steps', 'FontSize', 14)
    ylabel(ax2, 'Mean cos(heading − target bearing)', 'FontSize', 14)
    title(ax2, 'Mean Target Alignment', 'FontSize', 15, 'FontWeight', 'normal')

    frame = getframe(fig);
    writeVideo(writer, frame);
end

close(writer);
close(fig);
end

function X = vertcat_nonempty(C)
X = zeros(0, 2);
for i = 1:numel(C)
    if ~isempty(C{i})
        X = [X; C{i}]; %#ok<AGROW>
    end
end
end
