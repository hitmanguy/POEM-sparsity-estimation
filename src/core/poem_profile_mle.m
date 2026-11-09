function [kHat, varS, varI] = poem_profile_mle(sm, R, N, d, noiseVar, kLo, kHi, sigmaIFloor)
% POEM_PROFILE_MLE Joint sparsity-order and variance estimation via profile ML.
%   Implements Eq. (18) profile maximum likelihood over [kLo, kHi] with inner
%   Nelder-Mead optimization under Eq. (19) reparameterization and outer
%   coarse grid + local integer sweep.
%
% Inputs:
%   sm          - Per-row pooled measurement energy vector (Mx1 double)
%   R           - Number of pooled temporal rounds (1x1 integer)
%   N           - Ambient signal dimension (1x1 integer)
%   d           - Row degree of binary sensing matrix (1x1 integer)
%   noiseVar    - Calibrated sensor noise variance sigma_theta^2 (1x1 double)
%   kLo         - Lower search bracket bound k1 (1x1 integer)
%   kHi         - Upper search bracket bound k2 (1x1 integer)
%   sigmaIFloor - Background variance floor proxy from poem_bracket (1x1 double)
%
% Outputs:
%   kHat - Estimated sparsity order k_hat (1x1 integer)
%   varS - Estimated active coefficient variance sigma_s^2 (1x1 double)
%   varI - Estimated background coefficient variance sigma_i^2 (1x1 double)

    if ~isfinite(kLo) || ~isfinite(kHi) || kLo > kHi
        kHat = NaN;
        varS = NaN;
        varI = NaN;
        return;
    end

    sm = max(real(sm(:)), realmin);
    scale = max(mean(sm) / d, realmin);
    zz = sm / scale;
    nz = noiseVar / scale;
    options = optimset('Display', 'off', 'MaxIter', 90, 'MaxFunEvals', 180, 'TolX', 1e-4, 'TolFun', 1e-4);

    % Coarse grid evaluation
    coarse = unique(round(linspace(kLo, kHi, min(31, kHi - kLo + 1))));
    losses = inf(size(coarse));
    fitted = zeros(numel(coarse), 2);
    warm = [];
    initFloor = max(1e-5, (sigmaIFloor - noiseVar / d) / scale);

    for j = 1:numel(coarse)
        [losses(j), fitted(j, :)] = profile_at_k_G(coarse(j), zz, R, N, d, nz, initFloor, warm, options);
        warm = fitted(j, :);
    end
    [~, jBest] = min(losses);

    % Fine integer search around coarse mode (no parabolic refinement)
    rad = max(4, max(diff(coarse)));
    fine = max(kLo, coarse(jBest) - rad):min(kHi, coarse(jBest) + rad);
    fineLoss = inf(size(fine));
    fineFit = zeros(numel(fine), 2);
    warm = fitted(jBest, :);

    for j = 1:numel(fine)
        [fineLoss(j), fineFit(j, :)] = profile_at_k_G(fine(j), zz, R, N, d, nz, initFloor, warm, options);
        warm = fineFit(j, :);
    end
    [~, idx] = min(fineLoss);
    kHat = fine(idx);
    ubest = fineFit(idx, :);
    varI = scale * exp(ubest(1));
    varS = varI + scale * exp(ubest(2));
end

function [best, uBest] = profile_at_k_G(k, zz, R, N, d, nz, initFloor, warm, options)
    [~, h, lw] = hypergeom_weights(k, N, d);
    p = k / N;
    muPerCoeff = max((mean(zz) - nz) / d, 1e-5);
    vi0 = max(1e-5, min(initFloor, 0.95 * muPerCoeff));
    delta0 = max(1e-5, (muPerCoeff - vi0) / max(p, 1 / N));
    u0 = log([vi0, delta0]);
    fun = @(u) poem_composite_loglik(u, zz, R / 2, d, h, lw, nz);

    if ~isempty(warm) && all(isfinite(warm)) && fun(warm) < fun(u0)
        u0 = warm;
    end
    [uBest, best] = fminsearch(fun, u0, options);
    if ~isfinite(best) || any(~isfinite(uBest))
        best = fun(u0);
        uBest = u0;
    end
end
