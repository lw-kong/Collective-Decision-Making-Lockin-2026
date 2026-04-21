function [rpop, t_store, V_all, s_all, rateHz_all, acc] = func_run_LIF_2wang( ...
    A, k_in, G, V_init, noise_seed, syn_type, ...
    N, nsteps, dt, tau_m, V_L, V_th, V_reset, tau_ref, V_E, ...
    I0, beta, sigma_n, tau_ou, ...
    tau_AMPA, g_ratio_AMPA, tau_NMDA_decay, tau_NMDA_rise, alpha_NMDA, g_ratio_NMDA, ...
    Mg_conc, g_gap, omega, tau_r)
%
% Wang (2002)-style LIF network simulation.
%
% CHEMICAL SYNAPSE MODEL (key differences from original):
% =========================================================
%
%   1. CONDUCTANCE-BASED recurrent synapses:
%      Instead of:  I_rec = omega * J * sigmoid(s_avg)          [original]
%      We now use:  I_rec = omega * g * (V_E - V) * B(V) * s_avg  [Wang-style]
%
%      The driving force (V_E - V) is positive when V < V_E = 0 mV,
%      providing excitatory (depolarizing) current. Crucially, the
%      driving force DECREASES as V approaches V_E, providing a
%      natural self-limiting mechanism absent in current-based models.
%
%   2. VOLTAGE-DEPENDENT Mg2+ BLOCK (the biophysical nonlinearity):
%      B(V) = 1 / (1 + [Mg2+] * exp(-0.062*V) / 3.57)
%
%      This is the key change: the nonlinearity is NOT a hand-tuned
%      sigmoid on presynaptic activity, but emerges from the known
%      biophysics of NMDA receptors (Jahr & Stevens 1990).
%
%      At resting potential (V ~ -70 mV):  B ~ 0.04  (strongly blocked)
%      At threshold      (V ~ -50 mV):  B ~ 0.12  (partially unblocked)
%      At depolarized    (V ~ -20 mV):  B ~ 0.50  (substantially open)
%
%      This creates POSITIVE FEEDBACK:
%        high firing -> more NMDA current -> more depolarization
%        -> more Mg2+ relief -> even more NMDA current
%      which is the biophysical origin of the effective "quantization"
%      of the communication channel.
%
%   3. PROPER NMDA GATING with rise time and saturation:
%      dx_j/dt = -x_j / tau_rise  + delta(spike)
%      ds_j/dt = -s_j / tau_decay + alpha * x_j * (1 - s_j)
%
%      The (1 - s_j) saturation term prevents s from growing without
%      bound and is the standard biophysical model (Wang 2002).
%
%   4. SEPARATE AMPA + NMDA recurrent components:
%      Fast AMPA (tau ~ 2 ms): provides rapid excitatory drive
%      Slow NMDA (tau ~ 100 ms): provides slow, integrative drive
%      Both are conductance-based with reversal potential V_E = 0.
%
% GAP JUNCTION MODEL: unchanged from original (continuous voltage sharing).
%

rng(noise_seed);

% ---- Precomputed constants ----
dm      = dt / tau_m;                         % membrane Euler factor
dou     = exp(-dt / tau_ou);                  % OU noise decay per step
sig_ou  = sigma_n * sqrt(1 - dou^2);          % OU noise increment std
dr      = exp(-dt / tau_r);                   % rate-filter decay
ds_AMPA = exp(-dt / tau_AMPA);                % AMPA gating decay
dx_rise = exp(-dt / tau_NMDA_rise);           % NMDA rise variable decay

ksafe   = max(k_in, 1);                       % avoid division by zero
mask    = double(k_in > 0);                   % zero out unconnected

% ---- State variables ----
V       = V_init;                             % membrane potentials (mV)
xi      = zeros(N, 1);                        % OU noise state
ref     = zeros(N, 1);                        % refractory counters (ms)
rf      = zeros(N, 1);                        % rate filter (readout only)

% Synaptic gating variables (chemical synapse mode)
s_AMPA  = zeros(N, 1);    % fast AMPA gating variable
x_NMDA  = zeros(N, 1);    % NMDA rise variable
s_NMDA  = zeros(N, 1);    % slow NMDA gating variable (0 to 1, saturating)

% ---- Storage (1 ms resolution) ----
skip    = round(1 / dt);
nstore  = ceil(nsteps / skip);
rpop    = zeros(nstore, 1);
t_store = zeros(nstore, 1);
V_all   = zeros(N, nstore);
s_all   = zeros(N, nstore);
rateHz_all = zeros(N, nstore);
acc = zeros(N, nstore);
si      = 0;

% ---- Main simulation loop ----
for tt = 1 : nsteps

    % Ornstein-Uhlenbeck sensory noise
    xi = dou * xi + sig_ou * randn(N, 1);

    % Feedforward sensory current (current-based, in mV units)
    %   G=1: drive = I0 + beta + noise
    %   G=0: drive = I0 + noise
    Isens = I0 + beta * G(tt) + xi;

    % ============================================================
    % Recurrent current (depends on synapse type)
    % ============================================================
    switch syn_type

        case 'chemical'
            % ------------------------------------------------
            % WANG (2002)-STYLE CONDUCTANCE-BASED RECURRENT INPUT
            % ------------------------------------------------

            % Neighbor-averaged gating variables
            s_AMPA_avg = (A * s_AMPA) ./ ksafe;     % fast component
            s_NMDA_avg = (A * s_NMDA) ./ ksafe;     % slow component

            % ---- Voltage-dependent Mg2+ block ----
            % Jahr & Stevens (1990); Wang (2002) Eq. in Methods
            % B(V) -> 0 at hyperpolarized V (channel blocked)
            % B(V) -> 1 at depolarized V (channel open)
            B_Mg = 1.0 ./ (1.0 + (Mg_conc / 3.57) * exp(-0.062 * V));

            % ---- Conductance-based recurrent currents ----
            % I = g_ratio * (V_E - V) * gating
            % (V_E - V) > 0 for V < 0  =>  excitatory (depolarizing)
            %
            % Note: these enter the equation tau_m * dV/dt = -(V-VL) + Isens + Irec
            % with Irec in mV units (conductances are g/gL ratios).

            I_AMPA_rec = omega * g_ratio_AMPA * (V_E - V) .* s_AMPA_avg;
            I_NMDA_rec = omega * g_ratio_NMDA * (V_E - V) .* B_Mg .* s_NMDA_avg;

            Irec = mask .* (I_AMPA_rec + I_NMDA_rec);

        case 'gap'
            % ------------------------------------------------
            % GAP JUNCTIONS: unchanged from original
            % Linear voltage sharing, no nonlinearity.
            % I_gap = omega * g_gap * (<V>_neighbors - V_i)
            % Vanishes at consensus -> cannot sustain elevated activity.
            % ------------------------------------------------
            Irec = omega * g_gap * mask .* ...
                ((A * V) ./ ksafe - V);
    end

    % ============================================================
    % Membrane dynamics (forward Euler, refractory neurons frozen)
    % ============================================================
    active = ref <= 0;
    dV     = ( -(V - V_L) + Isens + Irec ) * dm;
    V      = V + dV .* active;

    % Spike detection
    spiked = active & (V >= V_th);

    % Post-spike reset and refractory period
    V(spiked)   = V_reset;
    ref(spiked) = tau_ref;
    ref         = ref - dt;

    % ============================================================
    % Synaptic gating variable updates (chemical synapse mode)
    % ============================================================
    if strcmp(syn_type, 'chemical')

        % ---- Fast AMPA gating ----
        % Simple exponential decay + spike-triggered increment
        % ds/dt = -s/tau_AMPA + delta(spike)
        s_AMPA         = s_AMPA * ds_AMPA;
        s_AMPA(spiked) = s_AMPA(spiked) + 1;

        % ---- Slow NMDA gating (Wang 2002 style) ----
        % Two-variable model:
        %   dx/dt = -x / tau_rise  + delta(spike)     [fast rise]
        %   ds/dt = -s / tau_decay + alpha * x * (1-s) [slow decay, saturating]
        %
        % The (1-s) term is crucial: it prevents s from exceeding 1,
        % providing biophysically realistic saturation.
        % At steady state with firing rate r:
        %   s_ss = alpha*r*tau_rise*tau_decay / (1 + alpha*r*tau_rise*tau_decay)
        %   e.g., r=20 Hz -> s_ss ~ 0.67;  r=40 Hz -> s_ss ~ 0.80

        % Rise variable: exponential decay + spike
        x_NMDA         = x_NMDA * dx_rise;
        x_NMDA(spiked) = x_NMDA(spiked) + 1;

        % Gating variable: forward Euler with saturation
        ds_NMDA_dt = -s_NMDA / tau_NMDA_decay + alpha_NMDA * x_NMDA .* (1 - s_NMDA);
        s_NMDA     = s_NMDA + dt * ds_NMDA_dt;

        % Clamp to [0, 1] for numerical safety
        s_NMDA = max(0, min(1, s_NMDA));
    end

    % Exponential rate filter for readout only (not part of dynamics)
    rf         = rf * dr;
    rf(spiked) = rf(spiked) + 1;

    % ---- Store at 1 ms intervals ----
    if mod(tt, skip) == 0
        si = si + 1;
        t_store(si) = tt * dt;

        rateHz   = rf / tau_r * 1000;
        rpop(si) = mean(rateHz);
        V_all(:, si) = V;
        rateHz_all(:, si) = rateHz;

        temp_decision = sign(rateHz - 25); % -1 or 1
        temp_decision = temp_decision/2+0.5; % 0 or 1
        acc(:, si) = 1 - abs(G(tt) - temp_decision);

        switch syn_type
            case 'chemical'
                s_all(:, si) = s_NMDA;   % store NMDA gating for analysis
            case 'gap'
                s_all(:, si) = V;
        end
    end
end

% Trim storage
rpop    = rpop(1:si);
t_store = t_store(1:si);
V_all   = V_all(:, 1:si);
s_all   = s_all(:, 1:si);
end