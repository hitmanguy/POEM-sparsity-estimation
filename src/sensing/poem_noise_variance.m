function v = poem_noise_variance(N, k, d, muS, varS, varI, snr_dB)
% POEM_NOISE_VARIANCE Sensor noise variance for a d-regular binary matrix at target SNR.
%   Implements calibrated noise variance sigma_theta^2 in Eq. (8) and Eq. (11).
%
% Inputs:
%   N      - Ambient signal dimension (1x1 integer)
%   k      - True sparsity order (1x1 integer)
%   d      - Row degree (1x1 integer)
%   muS    - Mean of active coefficients (1x1 double)
%   varS   - Variance of active coefficients sigma_s^2 (1x1 double)
%   varI   - Variance of background coefficients sigma_i^2 (1x1 double)
%   snr_dB - Target signal-to-noise ratio in dB (1x1 double)
%
% Outputs:
%   v      - Sensor noise variance sigma_theta^2 (1x1 double)

    p = k / N;
    eh = d * p;
    vh = d * p * (1 - p) * (N - d) / (N - 1);
    expectedPower = d * (p * varS + (1 - p) * varI) + muS^2 * (vh + eh^2);
    v = max(expectedPower * 10^(-snr_dB / 10), realmin);
end
