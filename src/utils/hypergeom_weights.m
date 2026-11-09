function [w, h, lw] = hypergeom_weights(k, N, d)
% HYPERGEOM_WEIGHTS Hypergeometric row-occupancy probabilities w_h(k).
%   Implements Eq. (10): w_h(k) = C(k,h)*C(N-k,d-h)/C(N,d).
%
% Inputs:
%   k - Sparsity order (1x1 integer, 1 <= k <= N-d)
%   N - Ambient signal dimension (1x1 integer)
%   d - Row degree of binary sensing matrix (1x1 integer)
%
% Outputs:
%   w  - Normalized occupancy weights (1xL vector, where L <= d+1)
%   h  - Feasible occupancy counts (1xL integer vector)
%   lw - Log-normalized occupancy weights (1xL vector)

    h = max(0, d - (N - k)):min(d, k);
    lw = log_choose(k, h) + log_choose(N - k, d - h) - log_choose(N, d);
    a = max(lw);
    w = exp(lw - a);
    lw = lw - (a + log(sum(w)));
    w = w / sum(w);
end
