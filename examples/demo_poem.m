% DEMO_POEM Minimal 25-round demonstration of the POEM sparsity estimator.

rng(42, 'twister');
N = 2500;       % Ambient signal dimension
k_true = 250;   % True sparsity order
M = 250;        % Number of binary sensing rows
d = 10;         % Row degree (ones per row)
R = 25;         % Number of temporal snapshots
snr_dB = 20;    % Sensor SNR (dB)

var_s = 400.0;  % Active coefficient variance
var_i = 2.34;   % Background coefficient variance (95% energy compaction)
noise_var = poem_noise_variance(N, k_true, d, 0, var_s, var_i, snr_dB);

S_supp = randperm(N, k_true);
B = poem_random_bsm(M, N, d);

Y = zeros(M, R);
for r = 1:R
    x = gen_synthetic_signal(N, S_supp, 0, var_s, var_i);
    Y(:, r) = B * x + sqrt(noise_var) * randn(M, 1);
end

[k_hat, sig_s2, sig_i2, info] = poem_estimate(Y, R, N, d, noise_var);

fprintf('True sparsity k      : %d\n', k_true);
fprintf('Search bracket [k1,k2]: [%d, %d]\n', info.k1, info.k2);
fprintf('POEM estimate k_hat  : %d (rel. error = %.2f%%)\n', k_hat, 100 * abs(k_hat - k_true) / k_true);
fprintf('Estimated variances  : sigma_s^2 = %.2f (true %.2f), sigma_i^2 = %.2f (true %.2f)\n', ...
    sig_s2, var_s, sig_i2, var_i);
