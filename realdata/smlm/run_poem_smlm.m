function results = run_poem_smlm(raw_smlm_dir)
% RUN_POEM_SMLM Reproduce Table IV SMLM sparsity estimation results (frames 2391-2400).
%   Evaluates POEM across rounds R = 1..10 for 100 trials and 20 trials (plus
%   single-run trajectory) from run_single_real_data.m against k_ref = 10.1 pixels.
%
% Inputs:
%   raw_smlm_dir - Optional path to SMLM folder (defaults to data/raw/sequence-MT0.N2.LD-2D-Exp-as-stack)
%
% Outputs:
%   results      - Struct with 100-trial, 20-trial, and single-run statistics

    t_start = tic;
    root_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    run(fullfile(root_dir, 'startup_poem.m'));

    % --- Configuration Block ---
    d = 60;
    numTrials_100 = 100;
    numTrials_20 = 20;
    seed_single = 100;
    seed_trials_base = 500;

    proc_file = fullfile(root_dir, 'data', 'processed', 'smlm_frames_2391_2400.mat');
    if ~isfile(proc_file)
        if nargin < 1, raw_smlm_dir = ''; end
        D = prepare_smlm(raw_smlm_dir);
    else
        D = load(proc_file);
    end

    X_dct = D.X_dct;
    N = D.N;
    M = round(N / 2);
    k_gt = D.k_gt;
    frame_seq = D.frame_seq;
    numRounds = numel(frame_seq);

    % 1. Single-run trajectory (seed = 100)
    rng(seed_single, 'twister');
    rows_b = repelem((1:M)', d);
    [~, idx_b] = sort(rand(M, N), 2);
    cols_b = reshape(idx_b(:, 1:d).', [], 1);
    B_single = sparse(rows_b, cols_b, 1, M, N);

    single_k_hat = zeros(numRounds, 1);
    s_total = zeros(M, 1);
    for r = 1:numRounds
        y = B_single * X_dct(:, r);
        s_total = s_total + y.^2;
        s_m = s_total / r;
        [k1, k2, vI_min_norm] = bounds_k1k2_smlm(s_m, r, N, d);
        single_k_hat(r) = estimate_g_smlm(s_m, r, N, d, k1, k2, vI_min_norm);
    end

    % 2. Multi-trial evaluation (t = 1..100; first 20 trials form the 20-trial block)
    kHat_trials = zeros(numTrials_100, numRounds);
    k1_trials = zeros(numTrials_100, numRounds);
    k2_trials = zeros(numTrials_100, numRounds);

    parfor t = 1:numTrials_100
        rng(seed_trials_base + t, 'twister');
        [~, idx_t] = sort(rand(M, N), 2);
        cols_t = reshape(idx_t(:, 1:d).', [], 1);
        B_t = sparse(rows_b, cols_t, 1, M, N);

        s_tot = zeros(M, 1);
        kh_row = zeros(1, numRounds);
        k1_row = zeros(1, numRounds);
        k2_row = zeros(1, numRounds);
        for r = 1:numRounds
            y = B_t * X_dct(:, r);
            s_tot = s_tot + y.^2;
            s_m = s_tot / r;
            [k1_t, k2_t, vI_min_t] = bounds_k1k2_smlm(s_m, r, N, d);
            kh_row(r) = estimate_g_smlm(s_m, r, N, d, k1_t, k2_t, vI_min_t);
            k1_row(r) = k1_t;
            k2_row(r) = k2_t;
        end
        kHat_trials(t, :) = kh_row;
        k1_trials(t, :) = k1_row;
        k2_trials(t, :) = k2_row;
    end

    fprintf('========================================================================================================\n');
    fprintf('SMLM MULTI-ROUND SPARSITY ESTIMATION (Table IV: Frames %d-%d, k_ref = %.1f px)\n', ...
        frame_seq(1), frame_seq(end), k_gt);
    fprintf('========================================================================================================\n');
    fprintf('%-5s | %-16s %-8s %-12s %-9s | %-10s %-16s %-9s\n', ...
        'Round', '100tr Mean+-SD', 'Median', 'Coverage(%)', 'MRE(%)', 'SingleRun', '20tr Mean+-SD', '20tr MRE(%)');
    fprintf('--------------------------------------------------------------------------------------------------------\n');

    summary_rows = zeros(numRounds, 9);
    for r = 1:numRounds
        kh100 = kHat_trials(:, r);
        mean100 = mean(kh100);
        sd100 = std(kh100);
        med100 = median(kh100);
        cov100 = 100 * mean(k1_trials(:, r) <= k_gt & k2_trials(:, r) >= k_gt);
        mre100 = 100 * mean(abs(kh100 - k_gt) / k_gt);

        kh20 = kHat_trials(1:numTrials_20, r);
        mean20 = mean(kh20);
        sd20 = std(kh20);
        mre20 = 100 * mean(abs(kh20 - k_gt) / k_gt);

        fprintf('%-5d | %6.1f +- %6.1f %-8.1f %-12.0f %-9.1f | %-10d %6.1f +- %6.1f %-9.1f\n', ...
            r, mean100, sd100, med100, cov100, mre100, single_k_hat(r), mean20, sd20, mre20);

        summary_rows(r, :) = [r, mean100, sd100, med100, cov100, mre100, single_k_hat(r), mean20, sd20];
    end
    fprintf('========================================================================================================\n');

    elapsed_sec = toc(t_start);
    out_dir = fullfile(root_dir, 'results');
    if ~isfolder(out_dir), mkdir(out_dir); end

    results = struct( ...
        'kHat_trials', kHat_trials, ...
        'k1_trials', k1_trials, ...
        'k2_trials', k2_trials, ...
        'single_k_hat', single_k_hat, ...
        'k_gt', k_gt, ...
        'summary', summary_rows, ...
        'matlab_version', version, ...
        'elapsed_sec', elapsed_sec);

    out_mat = fullfile(out_dir, 'poem_smlm_table4.mat');
    save(out_mat, '-struct', 'results');

    T = array2table(summary_rows, 'VariableNames', ...
        {'Round', 'Mean100', 'SD100', 'Median100', 'Coverage100_pct', 'MRE100_pct', 'SingleRun', 'Mean20', 'SD20'});
    writetable(T, fullfile(out_dir, 'poem_smlm_table4.csv'));

    fprintf('Saved results to %s\n', out_mat);
    fprintf('MATLAB version: %s | Elapsed time: %.2f s\n', version, elapsed_sec);
end

function [k1, k2, vI_min_norm] = bounds_k1k2_smlm(s, R, N, d)
    s = s(:);
    M_samples = numel(s);
    emp_var_y = mean(s);
    emp_second_s = mean(s.^2);
    mu_V_emp = emp_var_y / d;
    b_hat = emp_second_s / (1 + 2 / R);
    var_V_emp = max(b_hat - emp_var_y^2, 1e-4 * emp_var_y^2);
    se_var_V = sqrt((2 * b_hat^2 / M_samples) * (2 / R + 4 / R^2) + 4 * max(0, var_V_emp) * b_hat / (M_samples * R));
    var_V_hi = var_V_emp + 1.96 * se_var_V;
    fpc_factor = d * (N - d) / (N - 1);

    q_tail = 0.01;
    q_hi1 = min(0.05, q_tail + 1.96 * sqrt(q_tail * (1 - q_tail) / M_samples));
    g_q1 = (2 / R) * gammaincinv(q_tail, R / 2);
    sigma_i_k1 = (quantile(s, q_hi1) / g_q1) / d;

    if mu_V_emp > sigma_i_k1 && var_V_hi > 0
        k_mom_low = N / (1 + var_V_hi / (fpc_factor * (mu_V_emp - sigma_i_k1)^2));
        k1 = max(1, floor(k_mom_low));
    else
        k_mom_low = 1;
        k1 = 1;
    end

    q_nom = 0.15;
    q_lo = max(0.01, q_nom - 1.96 * sqrt(q_nom * (1 - q_nom) / M_samples));
    g_q2 = (2 / R) * gammaincinv(q_nom, R / 2);
    vI_min_norm = min(0.95, ((quantile(s, q_lo) / g_q2) / d) / mu_V_emp);

    hval = 0:d;
    pr_func = @(kk) (kk >= hval & (N - kk) >= (d - hval)) .* ...
        exp(log_choose(max(kk, hval), hval) + log_choose(max(N - kk, d - hval), d - hval) - log_choose(N, d));

    tau_s = (0.65^2) * emp_var_y;
    p_hat = sum(s <= tau_s) / M_samples;
    z_a = 1.96;
    P_small_adj = (p_hat + z_a^2 / (2 * M_samples) - z_a * sqrt(p_hat * (1 - p_hat) / M_samples + z_a^2 / (4 * M_samples^2))) / (1 + z_a^2 / M_samples);

    h_pos = hval(2:end);
    G0_eval = @(kk) sum(pr_func(kk) .* [1.0, gammainc((R / 2) * (0.65^2) * ((d * (kk / N)) ./ h_pos), R / 2)]);

    k_lo = k1;
    k_hi = N - d - 1;
    while (k_hi - k_lo) > 1
        k_mid = round((k_lo + k_hi) / 2);
        if G0_eval(k_mid) > P_small_adj, k_lo = k_mid; else, k_hi = k_mid; end
    end
    k2 = min(N - d - 1, max(max(k1 + 20, round(k_mom_low * 1.5)), k_hi));
end

function kHat = estimate_g_smlm(s, R, N, d, kLo, kHi, vI_min_norm)
    scale = mean(s) / d;
    z = s(:) / scale;
    mu_z = d;
    bz = mean(z.^2) / (1 + 2 / R);
    az = max(1e-2 * mu_z^2, bz - mu_z^2);

    lo_vec = [log(max(realmin, vI_min_norm)), log(realmin)];
    hi = log(1 / sqrt(eps));
    opt = optimset('Display', 'off', 'MaxIter', 80, 'MaxFunEvals', 160, 'TolX', 1e-4, 'TolFun', 1e-4);

    eval_map = containers.Map('KeyType', 'int32', 'ValueType', 'double');

    function nll_val = eval_cached(k_in)
        k_int = int32(k_in);
        if isKey(eval_map, k_int)
            nll_val = eval_map(k_int);
        else
            nll_val = fit_at_k_smlm(k_in, z, az, R, N, d, lo_vec, hi, opt);
            eval_map(k_int) = nll_val;
        end
    end

    a = kLo;
    b = kHi;
    while (b - a) > 12
        grid_log = round(exp(linspace(log(max(1, a)), log(b), 12)));
        grid_lin = round(linspace(a, b, 8));
        grid = unique([grid_log, grid_lin]);
        nll = inf(size(grid));
        for j = 1:numel(grid), nll(j) = eval_cached(grid(j)); end
        [~, jb] = min(nll);
        a = grid(max(1, jb - 1));
        b = grid(min(numel(grid), jb + 1));
    end

    grid = a:b;
    nll = inf(size(grid));
    for j = 1:numel(grid), nll(j) = eval_cached(grid(j)); end
    [~, jf] = min(nll);
    kf = grid(jf);
    sel = abs(grid - kf) <= 3 & isfinite(nll);
    k0 = kf;
    if nnz(sel) >= 3
        pq = polyfit(grid(sel) - kf, nll(sel), 2);
        if pq(1) > 0
            k0 = kf - pq(2) / (2 * pq(1));
        end
        k0 = min(max(k0, grid(1)), grid(end));
    end
    kHat = round(k0);
end

function best = fit_at_k_smlm(k, z, az, R, N, d, lo_vec, hi, opt)
    [~, h, logw] = hypergeom_weights(k, N, d);
    keep = logw > max(logw) - abs(log(eps));
    h = h(keep);
    logw = logw(keep);
    mx = max(logw);
    logw = logw - (mx + log(sum(exp(logw - mx))));

    pk = k / N;
    fpc = d * (N - d) / (N - 1);
    delta_mom = sqrt(az / max(fpc * pk * (1 - pk), realmin));
    vI_mom = max(exp(lo_vec(1)), 1 / (1 + pk * delta_mom / max(1 - pk * delta_mom, 0.15)));
    u_mom = log([vI_mom, max(exp(lo_vec(2)), (1 - min(0.99, vI_mom)) / pk)]);
    u_high = log([max(exp(lo_vec(1)), 0.82), 0.18 / pk]);
    fun = @(u) smlm_obj(u, z, R, d, h, logw, lo_vec, hi);
    if fun(u_high) < fun(u_mom)
        startU = u_high;
    else
        startU = u_mom;
    end
    [bestU, best] = fminsearch(fun, startU, opt);
    if ~(all(bestU >= lo_vec) && all(bestU <= hi))
        best = inf;
    end
end

function nll = smlm_obj(u, z, R, d, h, logw, lo_vec, hi)
    if any(~isfinite(u)) || any(u < lo_vec) || any(u > hi)
        nll = realmax;
        return;
    end
    V = d * exp(u(1)) + h * exp(u(2));
    A = (logw - 0.5 * R * log(2 * pi * V)) - (0.5 * R * z) * (1 ./ V);
    m = max(A, [], 2);
    nll = -sum(m + log(sum(exp(A - m), 2)));
    if ~isfinite(nll), nll = realmax; end
end
