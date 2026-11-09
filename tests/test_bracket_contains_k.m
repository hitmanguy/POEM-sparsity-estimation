function pass = test_bracket_contains_k()
% TEST_BRACKET_CONTAINS_K Verify [k1, k2] bracket encloses true k on a seeded case.
%   Tests poem_bracket (Eqs. 16-17) for N = 2500, k = 250, M = 250, d = 10, R = 25.
%
% Inputs:
%   None
%
% Outputs:
%   pass - Logical true if test passes (1x1 logical)

    rng(101, 'twister');
    N = 2500;
    k_true = 250;
    M = 250;
    d = 10;
    R = 25;
    varS = 400;
    varI = 2.339181286549708;
    noiseVar = poem_noise_variance(N, k_true, d, 0, varS, varI, 20);

    S = randperm(N, k_true);
    B = poem_random_bsm(M, N, d);

    e_sum = zeros(M, 1);
    for r = 1:R
        x = gen_synthetic_signal(N, S, 0, varS, varI);
        y = B * x + sqrt(noiseVar) * randn(M, 1);
        e_sum = e_sum + y.^2;
    end
    sm = e_sum / R;

    [k1, k2, dbg] = poem_bracket(sm, R, N, d);
    assert(~dbg.failed, 'Bracket failed unexpectedly.');
    assert(k1 >= 1 && k1 <= k_true && k_true <= k2 && k2 <= (N - d - 1), ...
        'Expected 1 <= k1 <= k_true <= k2 <= N-d-1, got [%d, %d] for k=%d.', k1, k2, k_true);

    pass = true;
    fprintf('[PASS] test_bracket_contains_k: k_true = %d inside [%d, %d]\n', k_true, k1, k2);
end
