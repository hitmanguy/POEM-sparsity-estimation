function [nll, loglik] = poem_composite_loglik(u, zz, shape, d, h, lw, nz)
% POEM_COMPOSITE_LOGLIK Composite marginal log-likelihood for scaled-chi^2 mixture.
%   Implements Eq. (14) composite marginal log-likelihood under the unconstrained
%   log-variance reparameterization of Eq. (19).
%
% Inputs:
%   u     - Log-variance parameters [u1, u2] (1x2 double)
%   zz    - Normalized per-row pooled energies sm / scale (Mx1 double)
%   shape - Gamma shape parameter R/2 (1x1 double)
%   d     - Row degree of binary sensing matrix (1x1 integer)
%   h     - Feasible support occupancy counts (1xL integer vector)
%   lw    - Log hypergeometric weights ln w_h(k) (1xL double vector)
%   nz    - Normalized sensor noise variance noiseVar / scale (1x1 double)
%
% Outputs:
%   nll    - Negative composite marginal log-likelihood (1x1 double)
%   loglik - Composite marginal log-likelihood l_comp (1x1 double)

    if any(~isfinite(u)) || any(u < -18) || any(u > 12)
        nll = 1e20;
        loglik = -nll;
        return;
    end
    vi = exp(u(1));
    delta = exp(u(2));
    V = max(realmin, d * vi + h * delta + nz);
    A = bsxfun(@plus, -shape * (zz(:) * (1 ./ V)), lw - shape * log(V));
    mx = max(A, [], 2);
    nll = -sum(mx + log(sum(exp(bsxfun(@minus, A, mx)), 2)));
    if ~isfinite(nll)
        nll = 1e20;
    end
    loglik = -nll;
end
