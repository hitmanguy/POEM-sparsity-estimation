function [kLo, kHi, dbg] = poem_bracket(sm, R, N, d)
% POEM_BRACKET Operational search bracket [k1, k2] for sparsity order k.
%   Implements Eq. (15) moment inversion, Eq. (16) lower bound k1 via delta-method
%   variance bound and 1% quantile floor, and Eq. (17) upper bound k2 via bisection
%   on the G0(k) sub-threshold envelope against the Wilson 95% lower bound.
%
% Inputs:
%   sm - Per-row pooled measurement energy vector (Mx1 double)
%   R  - Number of pooled temporal rounds (1x1 integer)
%   N  - Ambient signal dimension (1x1 integer)
%   d  - Row degree of binary sensing matrix (1x1 integer)
%
% Outputs:
%   kLo - Lower search bracket bound k1 (1x1 integer, unclipped in [1, N-d-1])
%   kHi - Upper search bracket bound k2 (1x1 integer, unclipped in [1, N-d-1])
%   dbg - Diagnostic struct (rawLower, rawUpper, varianceProxy, nan_fail, invert_fail, failed)

    sm = max(real(sm(:)), realmin);
    M = numel(sm);
    sbar = mean(sm);

    % Cross-row variance of conditional means (Eq. 15)
    bHat = mean(sm.^2) / (1 + 2 / R);
    vHat = bHat - sbar^2;
    Q = [sm, sm.^2];
    C = cov(Q, 1);
    grad = [-2 * sbar, 1 / (1 + 2 / R)];
    seV = sqrt(max(0, grad * C * grad' / M));
    vHi = max(0, vHat + 1.96 * seV);

    % Background floor proxy from lower tail (1% quantile, Eq. 16)
    sorted = sort(sm);
    pos = 1 + (M - 1) * 0.01;
    il = floor(pos);
    iu = ceil(pos);
    q01 = sorted(il) + (pos - il) * (sorted(iu) - sorted(il));
    g01 = (2 / R) * gammaincinv(0.01, R / 2);
    sigmaIFloor = q01 / (d * max(g01, realmin));

    % Moment inversion lower bound (Eq. 16)
    F = d * (N - d) / (N - 1);
    if sbar / d <= sigmaIFloor || vHi <= 0
        rawLower = 1;
    else
        denom = F * (sbar / d - sigmaIFloor)^2;
        rawLower = max(1, floor(N / (1 + vHi / max(denom, realmin))));
    end
    rawLower = min(N - d - 1, max(1, rawLower));

    % Upper bound via Wilson lower bound on empty rows (tau = 0.65^2 * sbar, Eq. 17)
    tau = (0.65^2) * sbar;
    pHat = mean(sm <= tau);
    z = 1.96;
    pWilson = (pHat + z^2 / (2 * M) - z * sqrt(pHat * (1 - pHat) / M + z^2 / (4 * M^2))) / (1 + z^2 / M);
    pWilson = max(0, min(1, pWilson));

    left = rawLower;
    right = N - d - 1;
    if left > right
        rawUpper = NaN;
    elseif envelope_G0(left, N, d, R) <= pWilson
        rawUpper = left;
    elseif envelope_G0(right, N, d, R) > pWilson
        rawUpper = right;
    else
        while right - left > 1
            mid = floor((left + right) / 2);
            if envelope_G0(mid, N, d, R) > pWilson
                left = mid;
            else
                right = mid;
            end
        end
        rawUpper = right;
    end

    k1 = max(1, rawLower);
    k2 = min(N - d - 1, rawUpper);

    is_nan_fail = ~isfinite(k1) || ~isfinite(k2);
    is_invert_fail = (k1 > k2);
    is_numerical_fail = is_nan_fail || is_invert_fail;

    if is_numerical_fail
        kLo = NaN;
        kHi = NaN;
    else
        kLo = k1;
        kHi = k2;
    end

    dbg = struct('rawLower', rawLower, ...
                 'rawUpper', rawUpper, ...
                 'varianceProxy', sigmaIFloor, ...
                 'nan_fail', is_nan_fail, ...
                 'invert_fail', is_invert_fail, ...
                 'failed', is_numerical_fail);
end

function g0 = envelope_G0(k, N, d, R)
    [w, h] = hypergeom_weights(k, N, d);
    q = ones(size(h));
    pos = h > 0;
    x = (R / 2) * (0.65^2) * d * k ./ (N * h(pos));
    q(pos) = gammainc(x, R / 2, 'lower');
    g0 = sum(w .* q);
end
