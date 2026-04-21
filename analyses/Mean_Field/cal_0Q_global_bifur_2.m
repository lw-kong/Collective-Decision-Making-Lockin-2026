% cal_0Q_scan_fixedpoints_ERk.m
% Find stable + unstable fixed points for quantized-message mean-field map
% using fsolve + multi-start, then classify stability by Jacobian eigenvalues.

%clear; clc;

%% ---------- parameters (same as your script) ----------
para_set = 0:0.01:5;          % social_weight scan

%Q = [-1,1];
Q = -10:10;

k = 6.82;
G = 0;
sigma_noise = 1 / sqrt(0.2);

sc = sigma_noise * sqrt(2*pi) / (2+sigma_noise * sqrt(2*pi));
omega_c = sigma_noise / k / sqrt(pi/2);

custom_thresholds = [];
Q = sort(Q(:).');
R = numel(Q);
if isempty(custom_thresholds)
    taus_mid = (Q(1:end-1) + Q(2:end)) / 2;
else
    taus_mid = custom_thresholds(:).';
    if numel(taus_mid) ~= R-1, error('custom_thresholds length must be numel(Q)-1'); end
end
taus = [-inf, taus_mid, inf];

%% ---------- multi-start initial guesses ----------
n_samples = 100*4;                 % increase to find more unstable roots
mu0_range = [-2, 2];
mu0_range = [-3.2, 3.2];
sigma0_range = [0.03, 1.5];

%% ---------- fsolve + classification options ----------
opts_fsolve = optimoptions('fsolve',...
    'Display','off',...
    'FunctionTolerance',1e-12,...
    'StepTolerance',1e-12,...
    'OptimalityTolerance',1e-12,...
    'MaxIterations',1000,...
    'MaxFunctionEvaluations',5000);

% Dedup tolerances for solutions
tol_mu = 1e-6;
tol_sg = 1e-6;

% Stability tolerance
tol_stab = 1e-8;

% Finite-diff step for Jacobian
eps_fd = 1e-6;

%% ---------- storage ----------
bifur_all(length(para_set)).para = [];
bifur_all(length(para_set)).mu_stable = [];
bifur_all(length(para_set)).mu_unstable = [];
bifur_all(length(para_set)).sigma_stable = [];
bifur_all(length(para_set)).sigma_unstable = [];
bifur_all(length(para_set)).eigvals = []; % cell optional

tic
for para_i = 1:length(para_set)
    social_weight = para_set(para_i);
    a = 1 / (k * social_weight + 1);

    % random initial guesses
    mu0_list = mu0_range(1) + rand(n_samples,1) * diff(mu0_range);
    sg0_list = sigma0_range(1) + rand(n_samples,1) * diff(sigma0_range);

    % collect roots found at this parameter
    roots_mu = [];
    roots_sg = [];
    roots_eigs = {};  %#ok<*AGROW>

    for s = 1:n_samples
        x0 = [mu0_list(s); sg0_list(s)];

        % solve F(x)-x = 0
        fun = @(x) fixedpoint_residual(x, a, G, sigma_noise, k, Q, taus);

        try
            [xsol, fval, exitflag] = fsolve(fun, x0, opts_fsolve);
        catch
            continue;
        end

        if exitflag <= 0, continue; end
        if any(~isfinite(xsol)), continue; end

        mu_sol = xsol(1);
        sg_sol = max(xsol(2), 1e-10); % avoid negative/zero sigma

        % check residual size
        if norm(fval) > 1e-7
            continue;
        end

        % deduplicate by clustering in (mu,sigma)
        [roots_mu, roots_sg, was_new] = add_clustered_root(roots_mu, roots_sg, mu_sol, sg_sol, tol_mu, tol_sg);
        if was_new
            % compute Jacobian and eigenvalues at this fixed point
            J = jacobian_fd(mu_sol, sg_sol, a, G, sigma_noise, k, Q, taus, eps_fd);
            ev = eig(J);
            roots_eigs{end+1,1} = ev;
        end
    end

    % classify roots by stability (discrete-time map)
    mu_st = [];
    sg_st = [];
    mu_un = [];
    sg_un = [];
    eigs_out = cell(numel(roots_mu),1);

    for r = 1:numel(roots_mu)
        ev = roots_eigs{r};
        eigs_out{r} = ev;
        if max(abs(ev)) < 1 - tol_stab
            mu_st(end+1,1) = roots_mu(r);
            sg_st(end+1,1) = roots_sg(r);
        else
            mu_un(end+1,1) = roots_mu(r);
            sg_un(end+1,1) = roots_sg(r);
        end
    end

    % store
    bifur_all(para_i).para = social_weight;
    bifur_all(para_i).mu_stable = sort(mu_st);
    bifur_all(para_i).sigma_stable = sg_st;
    bifur_all(para_i).mu_unstable = sort(mu_un);
    bifur_all(para_i).sigma_unstable = sg_un;
    bifur_all(para_i).eigvals = eigs_out;

    fprintf('sw = %.3f done. roots=%d (stable=%d, unstable=%d)\n', ...
        social_weight, numel(roots_mu), numel(mu_st), numel(mu_un));
    toc
end

%% ---------- plot: stable solid, unstable open ----------
% (use your colors if you like; here keep basic)
load('data_st_color_1.mat')

plot_global_ann_font_size = 13;
plot_global_label_font_size_math = 15;
ticks_font_size = 11;
plot_line_width = 1.3;

figure('Color','w'); hold on;

for para_i = 1:length(para_set)
    sw = bifur_all(para_i).para;

    mu_st = bifur_all(para_i).mu_stable;
    mu_un = bifur_all(para_i).mu_unstable;

    if ~isempty(mu_st)
        scatter(sw*ones(size(mu_st)), mu_st, 35, c_med_blue); % stable: filled
    end
    if ~isempty(mu_un)
        scatter(sw*ones(size(mu_un)), mu_un, 35, c_med_red);      % unstable: open circle
    end
end
%ylim([-1,1])
set(gca,'FontSize',ticks_font_size) % set gca before set labels
xlabel('Social Weight $\omega_s$','Interpreter','latex',...
    'FontSize',plot_global_label_font_size_math)
ylabel('Fixed Points of Mean Estimate','Interpreter','latex',...
    'FontSize',plot_global_label_font_size_math)
grid on; box on;
%title(sprintf('Fixed points (Q=[%s], G=%g, k=%.2f)', num2str(Q), G, k));
title(sprintf('Fixed points (Q=-1:0.5:1, G=%g, k=%.2f)',G, k));
%title(sprintf('Fixed points (Q=-10:1:10, G=%g, k=%.2f)',G, k));

%save('bif_fixedpoints_quantized.mat','bifur_all','para_set','Q','k','G','sigma_noise','taus');

%% ================= helper functions =================
function F = mf_map(mu, sg, a, G, sigma_noise, k, Q, taus)
    sg = max(sg, 1e-12);
    [c1, vQ] = quantizer_moments(mu, sg, Q, taus);
    mu_next = a*G + (1-a)*c1;
    sg_next = sqrt(a^2*sigma_noise^2 + (1-a)^2*vQ/k);
    F = [mu_next; sg_next];
end

function r = fixedpoint_residual(x, a, G, sigma_noise, k, Q, taus)
    mu = x(1); sg = x(2);
    F = mf_map(mu, sg, a, G, sigma_noise, k, Q, taus);
    r = F - [mu; max(sg, 1e-12)];
end

function J = jacobian_fd(mu, sg, a, G, sigma_noise, k, Q, taus, eps_fd)
    % finite difference Jacobian of the map F at (mu,sg)
    x = [mu; sg];
    F0 = mf_map(x(1), x(2), a, G, sigma_noise, k, Q, taus);

    J = zeros(2,2);
    for d = 1:2
        xp = x; xm = x;
        xp(d) = xp(d) + eps_fd;
        xm(d) = xm(d) - eps_fd;

        Fp = mf_map(xp(1), xp(2), a, G, sigma_noise, k, Q, taus);
        Fm = mf_map(xm(1), xm(2), a, G, sigma_noise, k, Q, taus);

        J(:,d) = (Fp - Fm) / (2*eps_fd);
    end
end

function [mu_list, sg_list, was_new] = add_clustered_root(mu_list, sg_list, mu, sg, tol_mu, tol_sg)
    was_new = true;
    if isempty(mu_list)
        mu_list = mu; sg_list = sg; return;
    end

    dmu = abs(mu_list - mu);
    dsg = abs(sg_list - sg);
    hit = find(dmu <= tol_mu & dsg <= tol_sg, 1, 'first');

    if isempty(hit)
        mu_list(end+1,1) = mu;
        sg_list(end+1,1) = sg;
    else
        was_new = false;
    end
end
