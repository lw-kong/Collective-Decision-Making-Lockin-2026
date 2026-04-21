%% run_hysteresis_G_2wang.m
% Hysteresis sweep for Wang-style LIF network.
% Control parameter: G (environmental drive), swept slowly:
%   branch-1: +2 -> -1
%   branch-2: -1 -> +2
%
% Readout: population firing rate rpop (Hz), plotted against G.



%% ======================== Parameters ==========================
% ---- Network geometry ----
Ngrid      = 8;
N          = Ngrid^2;
k_target   = 8;

% ---- LIF neuron biophysics (Wang-style) ----
Cm         = 0.5;        % nF
gL         = 0.025;      % uS
tau_m      = Cm / gL;    % ms
V_L        = -70;        % mV
V_th       = -50;        % mV
V_reset    = -55;        % mV
tau_ref    = 2;          % ms
V_E        = 0;          % mV

% ---- Timing ----
dt         = 0.05;       % ms
T_step     = 500;       % ms per G point (slow sweep)
tau_r      = 200;        % ms
tail_frac  = 0.4;        % average last 40% as quasi-steady output

% ---- Feedforward input ----
%I0         = 17.5;
%beta       = 5;
%sigma_n    = 0;
tau_ou     = 50;

I0         = 16;
beta       = 5;
sigma_n    = 1;
%tau_ou     = 20;

% ---- Recurrent synapses ----
tau_AMPA       = 2;
g_ratio_AMPA   = 0.05;
tau_NMDA_decay = 100;
tau_NMDA_rise  = 2;
alpha_NMDA     = 0.5;
g_ratio_NMDA   = 0.5;
Mg_conc        = 1.0;

% ---- Gap junction ----
g_gap      = 0.05;

% ---- Coupling strength ----
omega      = 2.5;

% ---- Which branch to run ----
% 'chemical' (default, nonlinear NMDA) or 'gap' (linear voltage sharing)
syn_type   = 'chemical';
repeat_num = 10;

% ---- Hysteresis sweep in G ----
G_max      = 1.6;
G_min      = -0.6;
n_G        = 80;%31;
G_down     = linspace(G_max, G_min, n_G);   % +2 -> -1
G_up       = linspace(G_min, G_max, n_G);   % -1 -> +2

G_down = G_max:-0.025:G_min;
G_up = G_min:0.025:G_max;

filename_save = ['save_hyster2wang_noise' num2str(sigma_n) ...
    '_omega' num2str(omega) '_' ...
    datestr(now,30) '_' num2str(randi(99)) '.mat'];

%% ======================== Build network =======================
A = generate_ring_network_more(N, k_target/2);
k_in = full(sum(A, 2));
fprintf('Network built: N = %d, <k> = %.1f, k in [%d, %d]\n', ...
    N, mean(k_in), min(k_in), max(k_in));

%% ======================== Packed config =======================
cfg = struct();
cfg.N               = N;
cfg.dt              = dt;
cfg.nsteps          = round(T_step / dt);
cfg.skip            = round(1 / dt);      % store every 1 ms
cfg.tail_frac       = tail_frac;

cfg.tau_m           = tau_m;
cfg.V_L             = V_L;
cfg.V_th            = V_th;
cfg.V_reset         = V_reset;
cfg.tau_ref         = tau_ref;
cfg.V_E             = V_E;

cfg.I0              = I0;
cfg.beta            = beta;
cfg.sigma_n         = sigma_n;
cfg.tau_ou          = tau_ou;
cfg.tau_r           = tau_r;

cfg.tau_AMPA        = tau_AMPA;
cfg.g_ratio_AMPA    = g_ratio_AMPA;
cfg.tau_NMDA_decay  = tau_NMDA_decay;
cfg.tau_NMDA_rise   = tau_NMDA_rise;
cfg.alpha_NMDA      = alpha_NMDA;
cfg.g_ratio_NMDA    = g_ratio_NMDA;
cfg.Mg_conc         = Mg_conc;

cfg.g_gap           = g_gap;
cfg.omega           = omega;

cfg.A               = A;
cfg.k_in            = k_in;
cfg.ksafe           = max(k_in, 1);
cfg.mask            = double(k_in > 0);

% Precomputed factors
cfg.dm              = dt / tau_m;
cfg.dou             = exp(-dt / tau_ou);
cfg.sig_ou          = sigma_n * sqrt(1 - cfg.dou^2);
cfg.dr              = exp(-dt / tau_r);
cfg.ds_AMPA         = exp(-dt / tau_AMPA);
cfg.dx_rise         = exp(-dt / tau_NMDA_rise);

%% ======================== Run sweep ===========================
tic
r_down_all = zeros(repeat_num, numel(G_down));
r_up_all   = zeros(repeat_num, numel(G_up));
area_loop_all = zeros(repeat_num, 1);
repeat_seed = zeros(repeat_num, 1);

for rr = 1:repeat_num
    seed_now = randi(1e9);
    repeat_seed(rr) = seed_now;
    rng(seed_now);

    state = init_state(cfg);

    fprintf('[%d/%d] Sweeping G: +2 -> -1 ...\n', rr, repeat_num);
    r_down = zeros(size(G_down));
    for i = 1:numel(G_down)
        [state, r_down(i)] = simulate_segment(state, G_down(i), syn_type, cfg);
    end

    fprintf('[%d/%d] Sweeping G: -1 -> +2 ...\n', rr, repeat_num);
    r_up = zeros(size(G_up));
    for i = 1:numel(G_up)
        [state, r_up(i)] = simulate_segment(state, G_up(i), syn_type, cfg);
    end

    % Loop area with both branches represented on increasing G grid
    r_down_inc = fliplr(r_down);      % map (+2->-1) to (-1->+2) order
    area_loop_all(rr) = trapz(G_up, abs(r_up - r_down_inc));

    r_down_all(rr, :) = r_down;
    r_up_all(rr, :)   = r_up;
    toc
end

fprintf('Estimated hysteresis loop area (mean +- std) = %.4f +- %.4f\n', ...
    mean(area_loop_all), std(area_loop_all));

%% ======================== Plot ================================
figure('Color', 'w', 'Position', [160 120 760 540]); hold on; box on;
for rr = 1:repeat_num
    plot(G_down, r_down_all(rr, :), '-o', 'LineWidth', 0.9, 'MarkerSize', 3, ...
        'Color', [0.10 0.35 0.85]);
    plot(G_up,   r_up_all(rr, :),   '-s', 'LineWidth', 0.9, 'MarkerSize', 3, ...
        'Color', [0.85 0.25 0.10]);
end
xlabel('G');
ylabel('Population rate r_{pop} (Hz)');
title(sprintf('Hysteresis loops (%s), repeats=%d, omega=%.1f',...
    syn_type, repeat_num, omega), ...
    'FontWeight', 'normal');
set(gca, 'FontSize', 11);

%% ======================== Save ================================
%save(filename_save);
fprintf('Saved: hysteresis_G_result_2wang.mat\n');

%% ======================== Local functions =====================
function state = init_state(cfg)
state.V      = cfg.V_L + 5 * randn(cfg.N, 1);
state.xi     = zeros(cfg.N, 1);
state.ref    = zeros(cfg.N, 1);
state.rf     = zeros(cfg.N, 1);
state.s_AMPA = zeros(cfg.N, 1);
state.x_NMDA = zeros(cfg.N, 1);
state.s_NMDA = zeros(cfg.N, 1);
end

function [state, r_tail] = simulate_segment(state, G_now, syn_type, cfg)
nstore = ceil(cfg.nsteps / cfg.skip);
r_store = zeros(nstore, 1);
si = 0;

for tt = 1:cfg.nsteps
    % OU noise
    state.xi = cfg.dou * state.xi + cfg.sig_ou * randn(cfg.N, 1);

    % Sweep-controlled environment drive
    Isens = cfg.I0 + cfg.beta * G_now + state.xi;

    % Recurrent term
    switch syn_type
        case 'chemical'
            s_AMPA_avg = (cfg.A * state.s_AMPA) ./ cfg.ksafe;
            s_NMDA_avg = (cfg.A * state.s_NMDA) ./ cfg.ksafe;

            B_Mg = 1.0 ./ (1.0 + (cfg.Mg_conc / 3.57) * exp(-0.062 * state.V));

            I_AMPA_rec = cfg.omega * cfg.g_ratio_AMPA * ...
                (cfg.V_E - state.V) .* s_AMPA_avg;
            I_NMDA_rec = cfg.omega * cfg.g_ratio_NMDA * ...
                (cfg.V_E - state.V) .* B_Mg .* s_NMDA_avg;

            Irec = cfg.mask .* (I_AMPA_rec + I_NMDA_rec);

        case 'gap'
            Irec = cfg.omega * cfg.g_gap * cfg.mask .* ...
                ((cfg.A * state.V) ./ cfg.ksafe - state.V);

        otherwise
            error('Unknown syn_type: %s', syn_type);
    end

    % LIF membrane dynamics
    active = state.ref <= 0;
    dV = (-(state.V - cfg.V_L) + Isens + Irec) * cfg.dm;
    state.V = state.V + dV .* active;

    % Spikes + reset + refractory
    spiked = active & (state.V >= cfg.V_th);
    state.V(spiked)   = cfg.V_reset;
    state.ref(spiked) = cfg.tau_ref;
    state.ref         = state.ref - cfg.dt;

    % Synaptic gating (only for chemical mode)
    if strcmp(syn_type, 'chemical')
        state.s_AMPA         = state.s_AMPA * cfg.ds_AMPA;
        state.s_AMPA(spiked) = state.s_AMPA(spiked) + 1;

        state.x_NMDA         = state.x_NMDA * cfg.dx_rise;
        state.x_NMDA(spiked) = state.x_NMDA(spiked) + 1;

        ds_NMDA_dt = -state.s_NMDA / cfg.tau_NMDA_decay + ...
            cfg.alpha_NMDA * state.x_NMDA .* (1 - state.s_NMDA);
        state.s_NMDA = state.s_NMDA + cfg.dt * ds_NMDA_dt;
        state.s_NMDA = max(0, min(1, state.s_NMDA));
    end

    % Rate readout
    state.rf = state.rf * cfg.dr;
    state.rf(spiked) = state.rf(spiked) + 1;

    if mod(tt, cfg.skip) == 0
        si = si + 1;
        rateHz = state.rf / cfg.tau_r * 1000;
        r_store(si) = mean(rateHz);
    end
end

r_store = r_store(1:si);
tail_n = max(1, round(cfg.tail_frac * numel(r_store)));
r_tail = mean(r_store(end - tail_n + 1 : end));
end
