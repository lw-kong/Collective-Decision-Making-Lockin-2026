%% Weighted-TopK(|.|) attention + Degroot-style average (mean-field map)
% Selection compares |omega_o * o| vs |omega_s * m|
% After selection, average uses same weights omega_o, omega_s


%% ---------------- Parameters ----------------
k = 8;                 % number of social inputs (d = k)
K = 4;                  % Top-K among (k social + 1 personal), must satisfy 1<=K<=k+1
omega_o = 1.0;          % personal weight
omega_s = 0.3;          % per-social-message weight
G = 1;                  % environment state (+1/-1)
sigma_noise = 1.0;      % o ~ N(G, sigma_noise^2)

step_max = 200;

% Social message distribution model: m_t ~ N(mu_t, sigma_t^2)
mu0 = 0.2;
sigma0 = 0.8;

% numerical integration over y_o = omega_o * o
n_grid = 4001;
grid_sd = 7;            % integrate y_o over mean +/- grid_sd*std

%% ---------------- Storage ----------------
mu_hist = nan(step_max,1);
si_hist = nan(step_max,1);
po_hist = nan(step_max,1);    % P(personal selected)
tau_hist = nan(step_max,1);   % threshold in y-space for social selection

mu = mu0;
si = max(sigma0, 1e-8);

mu_hist(1)=mu; si_hist(1)=si;

%% ---------------- Main iteration ----------------
for t = 2:step_max

    % ---- Work in y-space ----
    % y_o = omega_o * o ~ N(omega_o*G, (omega_o*sigma_noise)^2)
    mu_yo = omega_o * G;
    si_yo = abs(omega_o) * sigma_noise;

    % y_s = omega_s * m ~ N(omega_s*mu, (omega_s*si)^2)
    mu_ys = omega_s * mu;
    si_ys = abs(omega_s) * si;

    % 1) Personal selection probability p_o
    % Given u = |y_o|, personal is in Top-K iff
    %   #{j: |y_s,j| > u} <= K-1   where y_s,j i.i.d.
    % => p_sel(u) = BinomialCDF(K-1; k, p_tail(u)), p_tail(u)=P(|y_s|>u)
    % Then integrate over y_o distribution.
    yo_min = mu_yo - grid_sd*si_yo;
    yo_max = mu_yo + grid_sd*si_yo;
    yo_grid = linspace(yo_min, yo_max, n_grid);
    dyo = yo_grid(2)-yo_grid(1);

    fyo = normpdf(yo_grid, mu_yo, si_yo);
    u_grid = abs(yo_grid);

    p_tail_u = p_abs_tail_normal(u_grid, mu_ys, si_ys);  % P(|y_s|>u)
    p_sel_u  = binocdf(K-1, k, p_tail_u);

    p_o = sum(p_sel_u .* fyo) * dyo;

    % Conditional moments of y_o given selected
    if p_o < 1e-12
        mu_yo_sel = 0;
        var_yo_sel = si_yo^2;
    else
        mu_yo_sel = (sum(yo_grid .* p_sel_u .* fyo) * dyo) / p_o;
        Eyo2_sel  = (sum((yo_grid.^2) .* p_sel_u .* fyo) * dyo) / p_o;
        var_yo_sel = max(Eyo2_sel - mu_yo_sel^2, 0);
    end

    % Expected number of selected social messages
    K_s = K - p_o;
    K_s = max(min(K_s, K), 0);

    % 2) Determine tau (in y-space) so that k * P(|y_s|>tau) = K_s
    if K_s < 1e-10
        tau = inf;
        mu_ys_sel = 0;
        var_ys_sel = 0;
    else
        target_tail = K_s / k;

        tau_hi = abs(mu_ys) + 10*si_ys + 10;
        f = @(tau_) p_abs_tail_normal(tau_, mu_ys, si_ys) - target_tail;

        if f(0) < 0
            tau = 0;
        else
            while f(tau_hi) > 0
                tau_hi = tau_hi * 2;
                if tau_hi > 1e6, break; end
            end
            tau = fzero(f, [0, tau_hi]);
        end

        % 3) Moments of selected social y_s | |y_s|>tau
        [mu_ys_sel, var_ys_sel] = trunc_two_tail_moments(mu_ys, si_ys, tau);
    end

    % 4) Update mean and variance of e_{t+1}
    % e = ( I*y_o + sum_{l=1..K_s} y_s,l ) / ( omega_o*I + omega_s*K_s )
    denom = omega_o*p_o + omega_s*K_s;   % mean-field denom approximation

    if denom < 1e-12
        mu_next = 0;
        si_next = si;
    else
        % mean numerator
        num_mean = p_o*mu_yo_sel + K_s*mu_ys_sel;
        mu_next = num_mean / denom;

        % variance of numerator (approx)
        % Var(I*y_o) = p_o*Var(y_o|sel) + p_o*(1-p_o)*(E[y_o|sel])^2
        var_Iyo = p_o*var_yo_sel + p_o*(1-p_o)*mu_yo_sel^2;
        var_num = K_s*var_ys_sel + var_Iyo;

        si_next = sqrt(max(var_num,0)) / denom;
    end

    % store and roll
    mu = mu_next;
    si = max(si_next, 1e-8);

    mu_hist(t)=mu;
    si_hist(t)=si;
    po_hist(t)=p_o;
    tau_hist(t)=tau;

    if t>5 && max(abs([mu_hist(t)-mu_hist(t-1), si_hist(t)-si_hist(t-1)])) < 1e-10
        mu_hist = mu_hist(1:t);
        si_hist = si_hist(1:t);
        po_hist = po_hist(1:t);
        tau_hist = tau_hist(1:t);
        break;
    end
end

fprintf('Final: mu=%.6f, sigma=%.6f, steps=%d\n', mu_hist(end), si_hist(end), numel(mu_hist));

%% ---------------- Plots ----------------
figure('Color','w','Position',[100 100 720 420]);
subplot(2,2,1); plot(mu_hist,'o-'); grid on; xlabel('iter'); ylabel('\mu_t (message mean)');
subplot(2,2,2); plot(si_hist,'o-'); grid on; xlabel('iter'); ylabel('\sigma_t (message std)');
subplot(2,2,3); plot(po_hist,'o-'); grid on; xlabel('iter'); ylabel('p_o = P(personal selected)');
subplot(2,2,4); plot(tau_hist,'o-'); grid on; xlabel('iter'); ylabel('\tau_t (threshold in |y|)');

%% ---------------- Helper functions ----------------
function p = p_abs_tail_normal(u, mu, sigma)
% p = P(|X| > u) for X ~ N(mu, sigma^2), u>=0
u = max(u, 0);
sigma = max(sigma, eps);
a = (u - mu) ./ sigma;
b = (-u - mu) ./ sigma;
p = (1 - normcdf(a,0,1)) + normcdf(b,0,1);
end

function [m_sel, v_sel] = trunc_two_tail_moments(mu, sigma, tau)
% Moments of X | |X|>tau, where X ~ N(mu, sigma^2)
sigma = max(sigma, 1e-12);
tau = max(tau, 0);

alpha = (tau - mu)/sigma;
beta  = (-tau - mu)/sigma;

P_hi = 1 - normcdf(alpha,0,1);
P_lo = normcdf(beta,0,1);
P = P_hi + P_lo;

if P < 1e-14
    m_sel = 0;
    v_sel = 0;
    return;
end

phi_a = normpdf(alpha,0,1);
phi_b = normpdf(beta,0,1);

% Upper tail X>tau
lambda_u = phi_a / max(P_hi, 1e-14);
mean_u = mu + sigma * lambda_u;
var_u  = sigma^2 * (1 + alpha*lambda_u - lambda_u^2);

% Lower tail X<-tau
lambda_l = phi_b / max(P_lo, 1e-14);
mean_l = mu - sigma * lambda_l;
var_l  = sigma^2 * (1 - beta*lambda_l - lambda_l^2);

w_u = P_hi / P;
w_l = P_lo / P;

m_sel = w_u*mean_u + w_l*mean_l;

E2_u = var_u + mean_u^2;
E2_l = var_l + mean_l^2;
E2 = w_u*E2_u + w_l*E2_l;

v_sel = max(E2 - m_sel^2, 0);
end
