function v = log_choose(n, r)
% LOG_CHOOSE Natural logarithm of binomial coefficient n choose r.
%   Implements log-combinatorial terms for Eq. (10).
%
% Inputs:
%   n - Nonnegative integer scalar or array
%   r - Nonnegative integer scalar or array (same size as n or scalar)
%
% Outputs:
%   v - ln(n choose r), array matching size of broadcasted inputs

    v = gammaln(n + 1) - gammaln(r + 1) - gammaln(n - r + 1);
end
