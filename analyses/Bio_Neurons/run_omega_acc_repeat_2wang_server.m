%% run_omega_acc_repeat_2wang.m
% Sweep omega and evaluate mean(acc(:)) for chemical vs gap-junction modes.
% For each omega, run repeat_num independent repetitions.
% Plot curves of mean(acc_ch(:)) and mean(acc_gj(:)) versus omega.
par_num = 26;
parpool('local',par_num)

%% ======================== Parameters ==========================
% ---- Network geometry ----
Ngrid      = 8;
N          = Ngrid^2;
k_target   = 8;

% ---- LIF neuron biophysics (Wang 2002 style) ----
Cm         = 0.5;         % nF
gL         = 0.025;       % uS
tau_m      = Cm / gL;     % ms
V_L        = -70;         % mV
V_th       = -50;         % mV
V_reset    = -55;         % mV
tau_ref    = 2;           % ms
V_E        = 0;           % mV

% ---- Simulation timing ----
dt         = 0.05;        % ms
T_total    = 40000;       % ms
T_env      = 5000;        % ms
tau_r      = 200;         % ms

% ---- Sensory (feedforward) input ----
tau_ou     = 50;          % ms
I0         = 16;
beta       = 5;
sigma_n    = 2.5;

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

% ---- Omega sweep / repeats ----
%omega_list = 0:0.25:4;
omega_list = 0:0.25:6;
repeat_num = 50;

%% ======================== Build network =======================
%A = generate_ring_network_more(N, k_target/2);
A = generate_2d_open_lattice_8nn(N);
network_style = 'generate_2d_open_lattice_8nn';
k_in = full(sum(A, 2));
fprintf('Network built: N = %d, <k> = %.1f, k in [%d, %d]\n', ...
    N, mean(k_in), min(k_in), max(k_in));

%% ======================== Environment =========================
nsteps = round(T_total / dt);
tvec   = (0 : nsteps-1) * dt;
G = double(mod(tvec, 2*T_env) >= T_env);

%% ======================== Run sweep ===========================
n_omega = numel(omega_list);
mean_acc_ch = zeros(repeat_num, n_omega);
mean_acc_gj = zeros(repeat_num, n_omega);
seed_used   = zeros(repeat_num, n_omega);

fprintf('Running omega sweep: %d omegas x %d repeats ...\n', n_omega, repeat_num);
tic;
for rr = 1:repeat_num
    parfor io = 1:n_omega
        omega = omega_list(io);

        % Keep chemical/gap fair within the same (repeat, omega):
        % identical initial condition and identical noise seed.
        V_init = V_L + 5 * randn(N, 1);
        noise_seed = randi(1e9);
        seed_used(rr, io) = noise_seed;

        [~, ~, ~, ~, ~, acc_ch] = func_run_LIF_2wang( ...
            A, k_in, G, V_init, noise_seed, 'chemical', ...
            N, nsteps, dt, tau_m, V_L, V_th, V_reset, tau_ref, V_E, ...
            I0, beta, sigma_n, tau_ou, ...
            tau_AMPA, g_ratio_AMPA, tau_NMDA_decay, tau_NMDA_rise, alpha_NMDA, g_ratio_NMDA, ...
            Mg_conc, g_gap, omega, tau_r);

        [~, ~, ~, ~, ~, acc_gj] = func_run_LIF_2wang( ...
            A, k_in, G, V_init, noise_seed, 'gap', ...
            N, nsteps, dt, tau_m, V_L, V_th, V_reset, tau_ref, V_E, ...
            I0, beta, sigma_n, tau_ou, ...
            tau_AMPA, g_ratio_AMPA, tau_NMDA_decay, tau_NMDA_rise, alpha_NMDA, g_ratio_NMDA, ...
            Mg_conc, g_gap, omega, tau_r);

        mean_acc_ch(rr, io) = mean(acc_ch(:));
        mean_acc_gj(rr, io) = mean(acc_gj(:));

    end

    toc        
        fprintf('repeat %d/%d done\n', ...
            rr, repeat_num);
end
time_spent = toc;
fprintf('All done in %.1f s\n', time_spent);

%% ======================== Plot ================================
%{
figure('Color', 'w', 'Position', [120 100 850 560]); hold on; box on;
%{
for rr = 1:repeat_num
    plot(omega_list, mean_acc_ch(rr, :), '-o', 'LineWidth', 1.5, 'MarkerSize', 5, ...
        'Color', [0.10 0.35 0.85], ...
        'DisplayName', sprintf('chemical, repeat %d', rr));
    plot(omega_list, mean_acc_gj(rr, :), '-s', 'LineWidth', 1.5, 'MarkerSize', 5, ...
        'Color', [0.85 0.25 0.10], ...
        'DisplayName', sprintf('gap, repeat %d', rr));
end
%}
plot(omega_list, mean(mean_acc_ch,1), '-o', 'LineWidth', 1.5, 'MarkerSize', 5, ...
        'Color', [0.10 0.35 0.85], ...
        'DisplayName', sprintf('chemical, repeat %d', rr));
    plot(omega_list, mean(mean_acc_gj,1), '-s', 'LineWidth', 1.5, 'MarkerSize', 5, ...
        'Color', [0.85 0.25 0.10], ...
        'DisplayName', sprintf('gap, repeat %d', rr));
xlabel('\omega');
ylabel('mean(acc(:))');
title(sprintf('Accuracy vs \\omega (%d repeats)', repeat_num), 'FontWeight', 'normal');
legend('Location', 'best');
set(gca, 'FontSize', 11);
%}
%% ======================== Save ================================
save('omega_acc_repeat_2wang.mat');
fprintf('Saved: omega_acc_repeat_2wang.mat\n');
