%% run_wang_style.m
% ---------------------------------------------------------------
%  Wang (2002)-style biophysically realistic spiking network
%  for collective environmental tracking.
%
%  KEY CHANGES from run_0.m (original):
%    1. Conductance-based recurrent synapses (not current-based)
%    2. Voltage-dependent Mg2+ block of NMDA receptors (not sigmoid on s_avg)
%       -> The nonlinearity arises from biophysics, not from a hand-tuned sigmoid
%    3. Proper NMDA gating with rise time and saturation: ds/dt = -s/tau + alpha*x*(1-s)
%    4. Separate fast AMPA + slow NMDA recurrent components
%    5. Background Poisson excitatory input (mimicking cortical spontaneous activity)
%
%  The single self-activating excitatory group structure is preserved
%  (no competing groups, no inhibitory interneurons), consistent with
%  the environmental tracking framework in the main text.
%
%  Signal detection task: G(t) in {0, 1}
%  Two modes of communication compared:
%    1. Chemical synapses  -- conductance-based AMPA+NMDA with Mg block
%    2. Gap junctions      -- voltage-coded (continuous messages)
% ---------------------------------------------------------------

%% ======================== Parameters ========================

% ---- Network geometry ----
Ngrid      = 8;
N          = Ngrid^2;
lambda_c   = 1.5;
k_target   = 8;           % target number of neighbors

% ---- LIF neuron biophysics (Wang 2002, pyramidal cell values) ----
Cm         = 0.5;         % membrane capacitance (nF)
gL         = 0.025;       % leak conductance (uS)  [= 25 nS]
tau_m      = Cm / gL;     % membrane time constant (ms) [= 20 ms]
V_L        = -70;         % leak reversal potential (mV)
V_th       = -50;         % spike threshold (mV)
V_reset    = -55;         % reset potential after spike (mV)
tau_ref    = 2;           % absolute refractory period (ms)
V_E        = 0;           % excitatory reversal potential (mV)

% ---- Simulation timing ----
dt         = 0.05;        % integration time step (ms) [smaller for conductance-based]
T_total    = 40000;       % total simulation duration (ms)
T_env      = 5000;        % environment switch half-period (ms)

% ---- Sensory (feedforward) input ----
%  Kept current-based for simplicity and clean comparison with gap junctions.
%  In Wang (2002), external input is Poisson AMPA; here we use
%  baseline + stimulus + OU noise, as in the original code.
%I0         = 20.5;        % baseline drive (mV equiv.) -> ~20 Hz spontaneous
%I0 = 17.5;
%beta       = 2.0;         % stimulus strength (mV equiv.)
%beta = 5;
%sigma_n    = 0.6;         % OU noise amplitude (mV)

tau_ou     = 50;          % OU noise correlation time (ms)

I0 = 17;
beta = 5;
sigma_n = 1*2;

I0         = 16;
beta       = 5;
sigma_n    = 1;
% ---- Recurrent synaptic conductances ----
%  Defined as ratios to gL (dimensionless), so they enter the
%  tau_m * dV/dt equation directly in mV units.
%
%  Wang (2002) values (per synapse, for pyramidal cells):
%    g_ext_AMPA = 2.1 nS,  g_rec_AMPA = 0.05 nS,
%    g_NMDA = 0.165 nS,    gL = 25 nS
%  With ~240 recurrent synapses and w+ = 1.7:
%    total g_NMDA/gL ~ 0.165*240*1.7/25 ~ 2.7
%  We normalize by in-degree k, so effective ratio = omega * g_ratio_per_syn.

% Fast recurrent AMPA
tau_AMPA       = 2;       % AMPA decay time constant (ms)
g_ratio_AMPA   = 0.08;    % g_AMPA_total / gL (dimensionless), after k-normalization
                           % This is a relatively weak fast component
g_ratio_AMPA   = 0.05;
%g_ratio_AMPA = 0;

%g_ratio_AMPA   = 0.2;

% Slow recurrent NMDA
tau_NMDA_decay = 100;     % NMDA decay time constant (ms)  [Wang: 100 ms]
tau_NMDA_rise  = 2;       % NMDA rise time constant (ms)   [Wang: 2 ms]
alpha_NMDA     = 0.5;     % NMDA saturation rate (ms^-1)   [Wang: 0.5 ms^-1]
g_ratio_NMDA   = 0.35;    % g_NMDA_total / gL (dimensionless), after k-normalization
                           % This is the dominant recurrent component
%g_ratio_NMDA = 1;
g_ratio_NMDA   = 0.5;

%g_ratio_NMDA   = 0.05;

% Mg2+ block parameters
Mg_conc    = 1.0;         % extracellular [Mg2+] (mM)  [Wang: 1 mM]

% ---- Gap junction parameters ----
g_gap      = 0.05;        % gap junction coupling coefficient

% ---- Social weight (recurrent coupling strength) ----
omega      = 0;           % scales all recurrent conductances

% ---- Readout ----
tau_r      = 200;         % rate-estimation filter time constant (ms)

%% ======================== Build network =====================
A = generate_ring_network_more(N, k_target/2);
k_in = full(sum(A, 2));
fprintf('Network built: N = %d,  <k> = %.1f,  k in [%d, %d]\n', ...
    N, mean(k_in), min(k_in), max(k_in));

%% ======================== Environment =======================
nsteps = round(T_total / dt);
tvec   = (0 : nsteps-1) * dt;
G = double(mod(tvec, 2*T_env) >= T_env);

%% ======================== Initial conditions ================
V_init = V_L + 5 * randn(N, 1);    % start near leak potential

%% ======================== Simulate ==========================
noise_seed = randi(1e5);

fprintf('Simulating chemical synapses (Wang-style AMPA+NMDA) ... '); tic;
[rpop_ch, tst, V_all_ch, s_all_ch, rateHz_ch, acc_ch] = func_run_LIF_2wang( ...
    A, k_in, G, V_init, noise_seed, 'chemical', ...
    N, nsteps, dt, tau_m, V_L, V_th, V_reset, tau_ref, V_E, ...
    I0, beta, sigma_n, tau_ou, ...
    tau_AMPA, g_ratio_AMPA, tau_NMDA_decay, tau_NMDA_rise, alpha_NMDA, g_ratio_NMDA, ...
    Mg_conc, g_gap, omega, tau_r);
fprintf('done  (%.1f s)\n', toc);

fprintf('Simulating gap junctions (voltage-coded) ...  '); tic;
[rpop_gj, ~, V_all_gj, s_all_gj, rateHz_gj, acc_gj] = func_run_LIF_2wang( ...
    A, k_in, G, V_init, noise_seed, 'gap', ...
    N, nsteps, dt, tau_m, V_L, V_th, V_reset, tau_ref, V_E, ...
    I0, beta, sigma_n, tau_ou, ...
    tau_AMPA, g_ratio_AMPA, tau_NMDA_decay, tau_NMDA_rise, alpha_NMDA, g_ratio_NMDA, ...
    Mg_conc, g_gap, omega, tau_r);
fprintf('done  (%.1f s)\n', toc);

%% ======================== Plot ==============================
figure('Position', [100 100 1200 600]);

subplot(2,1,1);
hold on;
plot(tst/1000, rpop_ch, 'b', 'LineWidth', 1);
% Shade stimulus-ON periods
yL = ylim;
for k = 0 : floor(T_total / (2*T_env))
    t_on  = (2*k+1) * T_env / 1000;
    t_off = (2*k+2) * T_env / 1000;
    if t_on < T_total/1000
        patch([t_on min(t_off, T_total/1000) min(t_off, T_total/1000) t_on], ...
              [0 0 200 200], [1 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    end
end
plot(tst/1000, rpop_ch, 'b', 'LineWidth', 1);
xlabel('Time (s)'); ylabel('Pop. firing rate (Hz)');
title('Chemical synapses (Wang-style: conductance-based NMDA + Mg^{2+} block)');
ylim([0 max(rpop_ch)*1.3+1]);

subplot(2,1,2);
hold on;
for k = 0 : floor(T_total / (2*T_env))
    t_on  = (2*k+1) * T_env / 1000;
    t_off = (2*k+2) * T_env / 1000;
    if t_on < T_total/1000
        patch([t_on min(t_off, T_total/1000) min(t_off, T_total/1000) t_on], ...
              [0 0 200 200], [1 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    end
end
plot(tst/1000, rpop_gj, 'r', 'LineWidth', 1);
xlabel('Time (s)'); ylabel('Pop. firing rate (Hz)');
title('Gap junctions (voltage-coded, continuous messages)');
ylim([0 max(rpop_gj)*1.3+1]);

%%
ts = tst / 1000; 

figure('Color','w')
subplot(2,1,1)
imagesc(ts, 1:N, V_all_ch-V_reset)
colorbar
clim([0,5])

subplot(2,1,2)
imagesc(ts, 1:N, V_all_gj-V_reset)
colorbar
clim([0,5])


figure('Color','w')
subplot(2,1,1)
plot(ts, V_all_ch(1,:))
subplot(2,1,2)
plot(ts, V_all_ch(17,:))

figure('Color','w')
subplot(2,1,1)
plot(ts, s_all_ch)
subplot(2,1,2)
plot(ts, s_all_gj)

figure('Color','w')
subplot(2,1,1)
plot(ts, mean(acc_ch,1))
subplot(2,1,2)
plot(ts, mean(acc_gj,1))

plot_temp = rateHz_ch;
plot_temp1 = plot_temp(:,35010:40000);
plot_temp0 = plot_temp(:,30010:35000);
figure('Color','w')
subplot(2,1,1)
histogram(plot_temp1(:))
title(['omega=' num2str(omega) ', env = +1'])
subplot(2,1,2)
histogram(plot_temp0(:))
title('env = 0')

mean(acc_ch(:))
mean(acc_gj(:))