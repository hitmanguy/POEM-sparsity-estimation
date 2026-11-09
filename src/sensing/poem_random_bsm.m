function B = poem_random_bsm(M, N, d)
% POEM_RANDOM_BSM Generate static random d-regular binary sensing matrix.
%   Implements the sensing operator Phi_B in Eq. (8), drawing d distinct
%   column indices per row uniformly without replacement.
%
% Inputs:
%   M - Number of sensing rows m_B (1x1 integer)
%   N - Ambient signal dimension n (1x1 integer)
%   d - Row degree (ones per row, 1x1 integer, d <= N)
%
% Outputs:
%   B - Sparse binary matrix in {0,1}^{M x N} (MxN sparse double)

    rows = repelem((1:M)', d);
    cols = zeros(M * d, 1);
    for m = 1:M
        cols((m - 1) * d + (1:d)) = randperm(N, d);
    end
    B = sparse(rows, cols, 1, M, N);
end
