function [theta_out,a,T_cal] = func_ring_v3t(inputs, params)
% func_ring_sum  Integrate multiple directional cues with a 1-D ring-attractor.
%
%   theta_out = func_ring_sum(theta_strength)
%   theta_out = func_ring_sum(theta_strength, 'Name', Value, …)
%
% INPUT
%   theta_strength : k×2 array.  Col-1 = angles θ_i   (radians, 0–2π)
%                                  Col-2 = strengths  s_i > 0
%
% NAME–VALUE OPTIONS  (all optional)
%   'N',       256   – number of discrete neurons on the ring
%   'alpha',   1.0   – gain of recurrent input (cos kernel)
%   'beta',    1.0   – gain of external input
%   'tau',     0.2   – time constant (Euler update step size)
%   'Tmax',    200   – max iterations
%   'tol',   1e-6    – stop when max |Δa| < tol
%
% OUTPUT
%   theta_out : scalar angle (radians, 0–2π).  Center of final activity bump
%
% EXAMPLE
%   cues = [0.1  1.0;    % cue-1 at 0.1 rad, strength 1
%           3.0  0.8;    % cue-2 at 3.0 rad, strength 0.8
%           5.5  0.5];   % cue-3 at 5.5 rad, strength 0.5
%   theta_hat = func_ring_sum(cues);
%   fprintf('Integrated direction = %.2f rad (%.1f°)\n', ...
%           theta_hat, rad2deg(theta_hat));

% -------------------------------------------------------------------------
% Discretise ring ---------------------------------------------------------
num_neuron     = params.num_neuron;
phi_neurons   = linspace(0,2*pi,num_neuron+1)'; phi_neurons(end) = [];   % neuron preferred angles
a     = zeros(num_neuron,1);                             % initial activity
nu = params.nu;

% -------------------------------------------------------------------------
% Precompute recurrent weight matrix using cosine kernel ------------------
% W_ij = cos(Δθ.^ nu)  (periodic)
%dphi  = phi_neurons - phi_neurons.';
%W     = cos( abs(dphi/pi).^ nu *pi);                              % N×N dense matrix
% Normalize so that each column sums to zero (pure excitation+inhibition balance)
%W = W - mean(W(:));

dphi_circ = atan2(sin(phi_neurons - phi_neurons.'), cos(phi_neurons - phi_neurons.')); % V3
W = cos(abs(dphi_circ / pi) .^ nu * pi);
%W = W - mean(W(:));


% -------------------------------------------------------------------------
% External input: add each cue to nearest neuron (could smooth w/ Gaussian)
input_signal = zeros(num_neuron, 1);
for j = 1:size(inputs,1)

    delta = phi_neurons - inputs(j,1);
    delta = atan2(sin(delta), cos(delta)); 
    % still the same angles, but force delta into (-pi,pi]
    % so that at the center, we have delta = 0, which makes exp(-delta^2)
    % high
    input_signal = input_signal + ...
        abs(inputs(j,2)) * exp(-(delta.^2) / (2*params.sigma_exc^2));
end


% -------------------------------------------------------------------------
% Iterate until convergence ----------------------------------------------
%{
T_cal = params.Tmax;
for t_i = 1:params.Tmax
    a_old = a;
    a = ( 1 - params.factor_decay ) * a ...
        + params.factor_inter * W * tanh(a)/num_neuron ...
        + params.factor_input * input_signal;
    a(a < params.s_thres) = 0;
    if max(abs(a - a_old)) < params.tol
        T_cal = t_i;
        break
    end
end
%}

% extra parameters
% params.gamma, params.dt, params.tau
T_cal = params.Tmax;
for t_i = 1:params.Tmax
    rec_input = params.factor_inter * ...
        (W * tanh(params.gamma * a)) / num_neuron;
    ext_input = params.factor_input * input_signal;
    total_input = rec_input + ext_input;% - params.global_inhib * mean(a);
    da = (-a + max(total_input, 0)) * (params.dt / params.tau);
    a = a + da;
    if max(abs(da)) < params.tol
        T_cal = t_i;
        break
    end
end

    
    
    
    


% -------------------------------------------------------------------------
% Extract bump centre -----------------------------------------------------
if all(a==0)
    warning('Activity quenched to zero; returning NaN.');
    theta_out = NaN;
else
    % circular mean weighted by activity
    theta_out = atan2(sum(a .* sin(phi_neurons)), sum(a .* cos(phi_neurons)));
    if theta_out < 0, theta_out = theta_out + 2*pi; end
end
end
