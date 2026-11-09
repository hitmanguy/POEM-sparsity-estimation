function x = gen_synthetic_signal(N, S, muS, varS, varI)
% GEN_SYNTHETIC_SIGNAL Generate two-variance Gaussian signal on fixed support S.
%   Implements the two-variance signal prior in Eq. (7):
%   x_j ~ N(muS, varS) for j in S, and x_j ~ N(0, varI) for j not in S.
%
% Inputs:
%   N    - Ambient signal dimension (1x1 integer)
%   S    - Active support indices (1xk or kx1 integer vector)
%   muS  - Active coefficient mean (1x1 double)
%   varS - Active coefficient variance sigma_s^2 (1x1 double)
%   varI - Background coefficient variance sigma_i^2 (1x1 double)
%
% Outputs:
%   x    - Realization of compressible signal (Nx1 double)

    x = sqrt(varI) * randn(N, 1);
    x(S) = muS + sqrt(varS) * randn(numel(S), 1);
end
