function [k_hat, sig_s2, sig_i2, info] = poem_estimate(S, r, N, d, noiseVar)
% POEM_ESTIMATE Top-level POEM estimator for sparsity order and variances.
%   Implements Eqs. (15)-(19): constructs the operational search bracket [k1, k2]
%   from per-row pooled energies and maximizes the composite profile log-likelihood.
%
% Inputs:
%   S        - Per-row pooled energies s (Mx1 vector) OR raw measurement matrix Y (Mxr matrix)
%   r        - Number of pooled temporal rounds R (1x1 integer)
%   N        - Ambient signal dimension n (1x1 integer)
%   d        - Row degree of binary sensing matrix (1x1 integer)
%   noiseVar - Calibrated sensor noise variance sigma_theta^2 (1x1 double, optional, default 0)
%
% Outputs:
%   k_hat  - Estimated sparsity order (1x1 integer, or NaN if bracket fails)
%   sig_s2 - Estimated active coefficient variance sigma_s^2 (1x1 double)
%   sig_i2 - Estimated background coefficient variance sigma_i^2 (1x1 double)
%   info   - Struct with bracket bounds (k1, k2), boundary flag, and diagnostics

    if nargin < 5 || isempty(noiseVar)
        noiseVar = 0;
    end

    if size(S, 2) > 1
        r_use = min(r, size(S, 2));
        s = mean(abs(S(:, 1:r_use)).^2, 2);
    else
        s = S(:);
    end

    [k1, k2, dbg] = poem_bracket(s, r, N, d);

    if ~dbg.failed
        [k_hat, sig_s2, sig_i2] = poem_profile_mle(s, r, N, d, noiseVar, k1, k2, dbg.varianceProxy);
        boundary_hit = (k_hat == k1 || k_hat == k2);
    else
        k_hat = NaN;
        sig_s2 = NaN;
        sig_i2 = NaN;
        boundary_hit = false;
    end

    info = struct( ...
        'k1', k1, ...
        'k2', k2, ...
        'boundary_hit', boundary_hit, ...
        'failed', dbg.failed, ...
        'nan_fail', dbg.nan_fail, ...
        'invert_fail', dbg.invert_fail, ...
        'varianceProxy', dbg.varianceProxy);
end
