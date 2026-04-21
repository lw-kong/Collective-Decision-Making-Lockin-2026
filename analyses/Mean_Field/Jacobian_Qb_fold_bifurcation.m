%% Fold bifurcation workflow for Jacobian_Qb model
% One-shot script:
% 1) Find fixed points x* for each sw with fsolve + dedup
% 2) Build branch view (stable vs unstable) across sw
% 3) Locate fold candidate by solving [F=0, fx-1=0]
% 4) Plot three panels in one figure

%clear; clc;

%% Parameters (keep consistent with Jacobian_Qb.m)
G = 0;
k = 6.82;
sigma_n = 1.0 / sqrt(2 * 0.1);

%% Sweep setup
sw_min = 0.20;
sw_max = 0.80;
%sw_min = 0.50;
%sw_max = 1.10;
n_sw = 241;
sw_vec = linspace(sw_min, sw_max, n_sw);

% Initial guesses for scalar root search F(x,sw)=0
x_guess = linspace(-3.0, 2.0, 81);

% Numerical tolerances
root_tol = 1e-8;     % root acceptance: |F| < root_tol
dedup_tol = 1e-5;    % dedup roots within one sw
match_tol = 0.08;    % branch continuation distance across neighboring sw

%% fsolve options
opts1 = optimoptions('fsolve', ...
    'Display', 'off', ...
    'FunctionTolerance', 1e-12, ...
    'StepTolerance', 1e-12, ...
    'OptimalityTolerance', 1e-12);

opts2 = optimoptions('fsolve', ...
    'Display', 'off', ...
    'FunctionTolerance', 1e-14, ...
    'StepTolerance', 1e-14, ...
    'OptimalityTolerance', 1e-14, ...
    'MaxIterations', 400, ...
    'MaxFunctionEvaluations', 4000);


%% Step 1: fixed points + eigenvalues for each sw
roots_by_sw = cell(n_sw, 1);
lambda_by_sw = cell(n_sw, 1);
stable_by_sw = cell(n_sw, 1);

for i = 1:n_sw
    sw = sw_vec(i);
    candidates = nan(size(x_guess));
    kept = 0;

    for j = 1:numel(x_guess)
        x0 = x_guess(j);
        [x_sol, ~, exitflag] = fsolve(@(x) F_scalar(x, sw, G, k, sigma_n), x0, opts1);

        if exitflag <= 0
            continue;
        end

        if abs(F_scalar(x_sol, sw, G, k, sigma_n)) < root_tol
            kept = kept + 1;
            candidates(kept) = x_sol;
        end
    end

    if kept == 0
        roots_by_sw{i} = [];
        lambda_by_sw{i} = [];
        stable_by_sw{i} = [];
        continue;
    end

    roots = sort(candidates(1:kept));
    roots = dedup_sorted(roots, dedup_tol);

    lambda = arrayfun(@(x) fx_scalar(x, sw, G, k, sigma_n), roots);
    stable = abs(lambda) < 1;

    roots_by_sw{i} = roots;
    lambda_by_sw{i} = lambda;
    stable_by_sw{i} = stable;
end

%% Step 2: branch assembly (nearest-neighbor continuation)
[branch_sw, branch_x, branch_lambda, branch_stable] = ...
    build_branches(sw_vec, roots_by_sw, lambda_by_sw, stable_by_sw, match_tol);

%% Step 3: fold point candidate from system [F=0, fx-1=0]
% Use your prior observation as initial guess.
u0 = [-1.2; 0.8];  % [x; sw]

fold_fun = @(u) [ ...
    F_scalar(u(1), u(2), G, k, sigma_n); ...
    fx_scalar(u(1), u(2), G, k, sigma_n) - 1 ...
    ];

[u_fold, residual, exitflag_fold] = fsolve(fold_fun, u0, opts2);
x_fold = u_fold(1);
sw_fold = u_fold(2);
lambda_fold = fx_scalar(x_fold, sw_fold, G, k, sigma_n);

% Non-degeneracy checks for fold in F(x,sw)=0
hx = 1e-4;
hs = 1e-5;
Fxx = (F_scalar(x_fold + hx, sw_fold, G, k, sigma_n) ...
     - 2 * F_scalar(x_fold, sw_fold, G, k, sigma_n) ...
     + F_scalar(x_fold - hx, sw_fold, G, k, sigma_n)) / (hx^2);
Fsw = (F_scalar(x_fold, sw_fold + hs, G, k, sigma_n) ...
     - F_scalar(x_fold, sw_fold - hs, G, k, sigma_n)) / (2 * hs);

fprintf('\n=== Fold candidate from [F=0, fx-1=0] ===\n');
fprintf('exitflag        = %d\n', exitflag_fold);
fprintf('x_fold          = %.12g\n', x_fold);
fprintf('sw_fold         = %.12g\n', sw_fold);
fprintf('F(x_fold,sw)    = %.3e\n', residual(1));
fprintf('fx(x_fold,sw)-1 = %.3e\n', residual(2));
fprintf('lambda_fold     = %.12g\n', lambda_fold);
fprintf('F_xx (numeric)  = %.12g\n', Fxx);
fprintf('F_sw (numeric)  = %.12g\n', Fsw);


%% for plot panel C
eps_sw = 0.01;
sw_set = [sw_fold - eps_sw, sw_fold, sw_fold + eps_sw];
x_win = linspace(x_fold - 0.8, x_fold + 0.8, 801);

Fr_all = [];
for r = 1:numel(sw_set)
    sw_r = sw_set(r);
    Fr = arrayfun(@(x) F_scalar(x, sw_r, G, k, sigma_n), x_win);
    Fr_all = [Fr_all; Fr];
end


%save('save_BifurMapJacob_Qb_G0.mat');

%% Step 4: one-shot plotting (3 panels)
fig = figure('Color', 'w', 'Name', 'Fold bifurcation workflow');
tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel A: bifurcation diagram x* vs sw
nexttile; hold on; box on;
for b = 1:numel(branch_sw)
    s = branch_sw{b};
    x = branch_x{b};
    st = branch_stable{b};
    if numel(s) < 2
        continue;
    end

    for q = 1:(numel(s) - 1)
        if st(q) && st(q + 1)
            plot(s(q:q+1), x(q:q+1), 'b-', 'LineWidth', 1.8);
        elseif (~st(q)) && (~st(q + 1))
            plot(s(q:q+1), x(q:q+1), 'r--', 'LineWidth', 1.4);
        else
            % Stability changes between neighboring points; split style.
            plot(s(q), x(q), 'ko', 'MarkerSize', 3, 'LineWidth', 1.0);
            plot(s(q+1), x(q+1), 'ko', 'MarkerSize', 3, 'LineWidth', 1.0);
        end
    end
end
plot(sw_fold, x_fold, 'kp', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('sw');
ylabel('x^*');
title('Bifurcation diagram');
legend('stable', 'unstable', 'fold candidate', 'Location', 'best');
grid on;

% Panel B: eigenvalue along fixed-point branches
nexttile; hold on; box on;
for b = 1:numel(branch_sw)
    s = branch_sw{b};
    l = branch_lambda{b};
    st = branch_stable{b};
    if numel(s) < 2
        continue;
    end

    for q = 1:(numel(s) - 1)
        if st(q) && st(q + 1)
            plot(s(q:q+1), l(q:q+1), 'b-', 'LineWidth', 1.8);
        elseif (~st(q)) && (~st(q + 1))
            plot(s(q:q+1), l(q:q+1), 'r--', 'LineWidth', 1.4);
        else
            plot(s(q), l(q), 'ko', 'MarkerSize', 3, 'LineWidth', 1.0);
            plot(s(q+1), l(q+1), 'ko', 'MarkerSize', 3, 'LineWidth', 1.0);
        end
    end
end
yline(1, 'k--', 'LineWidth', 1.0);
yline(-1, 'k:', 'LineWidth', 1.0);
plot(sw_fold, lambda_fold, 'kp', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('sw');
ylabel('\lambda = f_x(x^*; sw)');
title('Jacobian eigenvalue on branches');
legend('stable branch', 'unstable branch', '\lambda=1', '\lambda=-1', ...
    'fold candidate', 'Location', 'best');
grid on;

% Panel C: tangency in F(x,sw)=f(x)-x near fold
nexttile; hold on; box on;


for r = 1:numel(sw_set)
    %sw_r = sw_set(r);
    %Fr = arrayfun(@(x) F_scalar(x, sw_r, G, k, sigma_n), x_win);
    plot(x_win, Fr_all(r,:), 'LineWidth', 1.8);
end
yline(0, 'k--', 'LineWidth', 1.0);
plot(x_fold, 0, 'kp', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('x');
ylabel('F(x,sw)=f(x,sw)-x');
title('Tangency near fold');
legend( ...
    sprintf('sw = sw_c - %.3g', eps_sw), ...
    'sw = sw_c', ...
    sprintf('sw = sw_c + %.3g', eps_sw), ...
    'F=0', ...
    'fold candidate', ...
    'Location', 'best');
grid on;

sgtitle(sprintf(['Fold workflow (G=%.2f, k=%.2f): ', ...
    'sw_c=%.6f, x_c=%.6f, lambda_c=%.6f'], ...
    G, k, sw_fold, x_fold, lambda_fold));

%% -------- Local functions --------
function y = F_scalar(x, sw, G, k, sigma_n)
    y = f_scalar(x, sw, G, k, sigma_n) - x;
end

function y = f_scalar(x, sw, G, k, sigma_n)
    omega_s = 1 - 1 / (k * sw + 1);
    a = 1 - omega_s;
    s = omega_s;

    m = 2 * normcdf(x, 0, 1) - 1;
    S = a^2 * sigma_n^2 + (s^2 / k) * (1 - m^2);
    D = sqrt(S);
    N = a * G + s * m;
    y = N / D;
end

function y = fx_scalar(x, sw, G, k, sigma_n)
    omega_s = 1 - 1 / (k * sw + 1);
    a = 1 - omega_s;
    s = omega_s;

    Phi = normcdf(x, 0, 1);
    phi = normpdf(x, 0, 1);
    m = 2 * Phi - 1;
    mp = 2 * phi;

    S = a^2 * sigma_n^2 + (s^2 / k) * (1 - m^2);
    D = sqrt(S);
    N = a * G + s * m;
    Np = s * mp;
    Sp = (s^2 / k) * (-2 * m * mp);
    Dp = 0.5 * Sp / sqrt(S);

    y = (Np * D - N * Dp) / (D^2);
end

function r = dedup_sorted(x_sorted, tol)
    if isempty(x_sorted)
        r = x_sorted;
        return;
    end

    r = x_sorted(1);
    for i = 2:numel(x_sorted)
        if abs(x_sorted(i) - r(end)) > tol
            r(end + 1) = x_sorted(i); %#ok<AGROW>
        else
            % Average close roots to reduce seed dependence.
            r(end) = 0.5 * (r(end) + x_sorted(i));
        end
    end
end

function [bsw, bx, bl, bst] = build_branches(sw_vec, roots_by_sw, lambda_by_sw, stable_by_sw, match_tol)
    n_sw = numel(sw_vec);

    % Branch struct array with fields:
    % sw, x, lambda, stable, last_x, active
    branches = struct('sw', {}, 'x', {}, 'lambda', {}, ...
        'stable', {}, 'last_x', {}, 'active', {});

    for i = 1:n_sw
        sw = sw_vec(i);
        xr = roots_by_sw{i};
        lr = lambda_by_sw{i};
        sr = stable_by_sw{i};

        if isempty(xr)
            for b = 1:numel(branches)
                branches(b).active = false;
            end
            continue;
        end

        used = false(size(xr));

        % Try to continue existing active branches first.
        for b = 1:numel(branches)
            if ~branches(b).active
                continue;
            end
            [dmin, idx] = min(abs(xr - branches(b).last_x));
            if (~used(idx)) && (dmin <= match_tol)
                branches(b).sw(end + 1) = sw;
                branches(b).x(end + 1) = xr(idx);
                branches(b).lambda(end + 1) = lr(idx);
                branches(b).stable(end + 1) = sr(idx);
                branches(b).last_x = xr(idx);
                used(idx) = true;
            else
                branches(b).active = false;
            end
        end

        % Start new branches for unmatched roots.
        for j = 1:numel(xr)
            if used(j)
                continue;
            end
            nb.sw = sw;
            nb.x = xr(j);
            nb.lambda = lr(j);
            nb.stable = sr(j);
            nb.last_x = xr(j);
            nb.active = true;
            branches(end + 1) = nb; %#ok<AGROW>
        end
    end

    % Keep branches with enough points for plotting continuity.
    keep = false(1, numel(branches));
    for b = 1:numel(branches)
        keep(b) = numel(branches(b).sw) >= 3;
    end
    branches = branches(keep);

    bsw = cell(numel(branches), 1);
    bx = cell(numel(branches), 1);
    bl = cell(numel(branches), 1);
    bst = cell(numel(branches), 1);
    for b = 1:numel(branches)
        bsw{b} = branches(b).sw;
        bx{b} = branches(b).x;
        bl{b} = branches(b).lambda;
        bst{b} = branches(b).stable;
    end
end
