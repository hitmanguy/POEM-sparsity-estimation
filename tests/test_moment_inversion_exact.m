function pass = test_moment_inversion_exact()
% TEST_MOMENT_INVERSION_EXACT Verify Proposition 1 recovers true k from exact moments.
%   Tests Eq. (15) closed-form moment inversion across multiple (N, k, d, sigma_s^2,
%   sigma_i^2, sigma_theta^2) parameter combinations.
%
% Inputs:
%   None
%
% Outputs:
%   pass - Logical true if all moment inversions match k within 1e-12 (1x1 logical)

    cases = [ ...
        2500, 250, 10, 400.0, 2.0,  1.5; ...
        5000, 100, 16,  25.0, 0.5,  0.1; ...
        5000, 600,  8,   5.0, 0.1,  0.0; ...
        1024,  15, 10,  50.0, 0.2,  0.05 ...
    ];

    for i = 1:size(cases, 1)
        N = cases(i, 1);
        k_true = cases(i, 2);
        d = cases(i, 3);
        sig_s2 = cases(i, 4);
        sig_i2 = cases(i, 5);
        sig_v2 = cases(i, 6);

        Delta = sig_s2 - sig_i2;
        F = d * (N - d) / (N - 1);

        % Population moments (Eq. 15)
        mu = d * sig_i2 + sig_v2 + (d * k_true / N) * Delta;
        v = (Delta^2) * F * (k_true / N) * (1 - k_true / N);

        % Exact Proposition 1 closed-form inversion on population moments (Eq. 15)
        k_exact = N / (1 + v / (F * ((mu - sig_v2) / d - sig_i2)^2));
        assert(abs(k_exact - k_true) / k_true < 1e-13, ...
            'Exact moment inversion error on case %d: got %.15f, expected %d.', i, k_exact, k_true);

        % Verify against hypergeometric mixture weights (gammaln precision ~ N*eps)
        [w, h] = hypergeom_weights(k_true, N, d);
        V_h = d * sig_i2 + sig_v2 + h * Delta;
        mu_mix = sum(w .* V_h);
        v_mix = sum(w .* (V_h - mu_mix).^2);

        assert(abs(mu - mu_mix) / max(abs(mu), 1) < 1e-9, 'First moment mismatch on case %d.', i);
        assert(abs(v - v_mix) / max(abs(v), 1) < 1e-9, 'Second central moment mismatch on case %d.', i);

        k_rec = N / (1 + v_mix / (F * ((mu_mix - sig_v2) / d - sig_i2)^2));
        assert(abs(k_rec - k_true) / k_true < 1e-9, ...
            'Moment inversion error on case %d: got %.12f, expected %d.', i, k_rec, k_true);
    end

    pass = true;
    fprintf('[PASS] test_moment_inversion_exact: Proposition 1 exact across %d configurations\n', size(cases, 1));
end
