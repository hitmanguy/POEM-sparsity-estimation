function results = run_poem_deepmimo(raw_mat_path)
% RUN_POEM_DEEPMIMO Reproduce Table III DeepMIMO sparsity estimation results.
%   Evaluates POEM over 100 trials on DeepMIMO complex channel snapshots
%   (N = 1024, M = 1024, d = 100, sliding window n_win = 10) against the
%   95%-energy reference k_0.95 using Eqs. (20)-(21) for MRE and MAPE.
%
% Inputs:
%   raw_mat_path - Optional path to cir.mat (defaults to data/raw/cir.mat)
%
% Outputs:
%   results      - Struct with per-round estimates, brackets, and Table III metrics

    t_start = tic;
    root_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    run(fullfile(root_dir, 'startup_poem.m'));

    % --- Configuration Block ---
    checkpoints = [5, 10, 20, 40, 50];
    trials = 100;
    M = 1024;
    d = 100;
    Wwin = 10;
    seed_mult = 10007;
    seed_offset = 7919;

    proc_file = fullfile(root_dir, 'data', 'processed', 'deepmimo_snapshots.mat');
    if nargin >= 1 && ~isempty(raw_mat_path)
        D = prepare_deepmimo(raw_mat_path);
    elseif ~isfile(proc_file)
        D = prepare_deepmimo();
    else
        D = load(proc_file);
        if ~isfield(D, 'P')
            D = prepare_deepmimo();
        end
    end

    X = D.X;
    P = D.P;
    N = D.N;
    totalRounds = D.totalRounds;
    nCp = numel(checkpoints);

    khatAd = nan(trials, nCp);
    vsAd = nan(trials, nCp);
    viAd = nan(trials, nCp);
    k1Ad = nan(trials, nCp);
    k2Ad = nan(trials, nCp);

    parfor t = 1:trials
        rng(seed_mult * t + seed_offset, 'twister');
        B = poem_random_bsm(M, N, d);
        Ebuf = zeros(M, totalRounds);

        kA = nan(1, nCp);
        vsA = nan(1, nCp);
        viA = nan(1, nCp);
        k1A = nan(1, nCp);
        k2A = nan(1, nCp);

        for r = 1:totalRounds
            x = X(r, :).';
            y = B * x;
            y2 = abs(y).^2;
            Ebuf(:, r) = y2;

            ci = find(checkpoints == r, 1);
            if ~isempty(ci)
                lo = max(1, r - Wwin + 1);
                s_ad = mean(Ebuf(:, lo:r), 2);
                nUse = r - lo + 1;
                Reff = 2 * nUse;

                [bLo, bHi] = bounds_k1k2_deepmimo(s_ad, Reff, N, d);
                kLo = bLo;
                kHi = max(bHi, kLo + 1);

                [kA(ci), vsA(ci), viA(ci)] = estimate_g_deepmimo(s_ad, Reff, N, d, kLo, kHi);
                k1A(ci) = bLo;
                k2A(ci) = bHi;
            end
        end

        khatAd(t, :) = kA;
        vsAd(t, :) = vsA;
        viAd(t, :) = viA;
        k1Ad(t, :) = k1A;
        k2Ad(t, :) = k2A;
    end

    fprintf('========================================================================================\n');
    fprintf('DEEPMIMO SPARSITY ESTIMATION RESULTS (Table III: N = %d, M = %d, d = %d, Trials = %d)\n', ...
        N, M, d, trials);
    fprintf('========================================================================================\n');
    fprintf('%-4s | %-5s | %-7s | %-16s | %-8s | %-8s | %-15s | %-12s\n', ...
        'R', 'n_win', 'k_0.95', 'k_hat (mean+-SD)', 'MRE (%)', 'MAPE (%)', 'Bracket [k1,k2]', 'Coverage (%)');
    fprintf('----------------------------------------------------------------------------------------\n');

    k95f = @(p) find(cumsum(sort(p, 'descend')) / sum(p) >= 0.95, 1);
    summary_rows = zeros(nCp, 10);

    for ci = 1:nCp
        r = checkpoints(ci);
        lo = max(1, r - Wwin + 1);
        n_win = r - lo + 1;
        kRefProf = k95f(mean(P(lo:r, :), 1));
        kh = khatAd(:, ci);
        b1 = mean(k1Ad(:, ci));
        b2 = mean(k2Ad(:, ci));
        inProf = 100 * mean(k1Ad(:, ci) <= kRefProf & kRefProf <= k2Ad(:, ci));
        errProf = kh - kRefProf;
        mean_k = mean(kh);
        sd_k = std(kh);
        mre = 100 * mean(errProf) / kRefProf;
        mape = 100 * mean(abs(errProf)) / kRefProf;

        fprintf('%-4d | %-5d | %7.1f | %6.2f +- %5.2f  | %+7.1f  | %7.1f  | [%4.1f, %5.1f]   | %11.0f\n', ...
            r, n_win, kRefProf, mean_k, sd_k, mre, mape, b1, b2, inProf);

        summary_rows(ci, :) = [r, n_win, kRefProf, mean_k, sd_k, mre, mape, b1, b2, inProf];
    end
    fprintf('========================================================================================\n');

    elapsed_sec = toc(t_start);
    out_dir = fullfile(root_dir, 'results');
    if ~isfolder(out_dir), mkdir(out_dir); end

    results = struct( ...
        'khatAd', khatAd, ...
        'k1Ad', k1Ad, ...
        'k2Ad', k2Ad, ...
        'summary', summary_rows, ...
        'matlab_version', version, ...
        'elapsed_sec', elapsed_sec);

    out_mat = fullfile(out_dir, 'poem_deepmimo_table3.mat');
    save(out_mat, '-struct', 'results');

    T = array2table(summary_rows, 'VariableNames', ...
        {'R', 'n_win', 'k_095', 'Mean_k_hat', 'SD_k_hat', 'MRE_pct', 'MAPE_pct', 'Mean_k1', 'Mean_k2', 'Coverage_pct'});
    writetable(T, fullfile(out_dir, 'poem_deepmimo_table3.csv'));

    fprintf('Saved results to %s\n', out_mat);
    fprintf('MATLAB version: %s | Elapsed time: %.2f s\n', version, elapsed_sec);
end

function [k1, k2] = bounds_k1k2_deepmimo(s, R, N, d)
    s = s(:);
    M_samples = numel(s);
    emp_var_y = mean(s);
    emp_second_s = mean(s.^2);
    mu_V_emp = emp_var_y / d;

    b_hat = emp_second_s / (1 + 2 / R);
    var_V_emp = b_hat - emp_var_y^2;
    se_var_V = sqrt((2 * b_hat^2 / M_samples) * (2 / R + 4 / R^2) + 4 * max(0, var_V_emp) * b_hat / (M_samples * R));
    var_V_hi = var_V_emp + 1.96 * se_var_V;
    fpc_factor = d * (N - d) / (N - 1);

    q_tail = 0.01;
    q_hi1 = min(0.05, q_tail + 1.96 * sqrt(q_tail * (1 - q_tail) / M_samples));
    g_q = (2 / R) * gammaincinv(q_tail, R / 2);
    sigma_i_est2 = (quantile(s, q_hi1) / g_q) / d;

    if mu_V_emp > sigma_i_est2 && var_V_hi > 0
        k_mom_low = N / (1 + var_V_hi / (fpc_factor * (mu_V_emp - sigma_i_est2)^2));
        k1 = max(1, floor(k_mom_low));
    else
        k1 = 1;
    end

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
        if G0_eval(k_mid) > P_small_adj
            k_lo = k_mid;
        else
            k_hi = k_mid;
        end
    end
    k2 = min(N - d - 1, max(max(k1 + 25, round(k_hi * 2.5)), k_hi));
end

function [kHat, vS, vI] = estimate_g_deepmimo(s, R, N, d, kLo, kHi)
    scale = mean(s) / d;
    z = s(:) / scale;
    mu_z = d;
    bz = mean(z.^2) / (1 + 2 / R);
    az = max(1e-4 * mu_z^2, bz - mu_z^2);

    lo = log(realmin);
    hi = log(1 / sqrt(eps));
    opt = optimset('Display', 'off', 'MaxIter', 80, 'MaxFunEvals', 160, 'TolX', 1e-4, 'TolFun', 1e-4);
    a = kLo;
    b = kHi;
    u = [];
    while (b - a) > 12
        grid = unique(round(linspace(a, b, 12)));
        nll = inf(size(grid));
        U = nan(numel(grid), 2);
        for j = 1:numel(grid)
            [nll(j), u] = fit_at_k_deepmimo(grid(j), z, az, R, N, d, u, lo, hi, opt);
            U(j, :) = u;
        end
        [~, jb] = min(nll);
        a = grid(max(1, jb - 1));
        b = grid(min(numel(grid), jb + 1));
        u = U(jb, :);
    end
    grid = a:b;
    nll = inf(size(grid));
    U = nan(numel(grid), 2);
    for j = 1:numel(grid)
        [nll(j), u] = fit_at_k_deepmimo(grid(j), z, az, R, N, d, u, lo, hi, opt);
        U(j, :) = u;
    end
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
    idxHat = find(grid == kHat, 1);
    if ~isempty(idxHat)
        uBest = U(idxHat, :);
    else
        [~, uBest] = fit_at_k_deepmimo(kHat, z, az, R, N, d, U(jf, :), lo, hi, opt);
    end
    vI = exp(uBest(1)) * scale;
    vS = vI + exp(uBest(2)) * scale;
end

function [best, bestU] = fit_at_k_deepmimo(k, z, az, R, N, d, u0, lo, hi, opt)
    [~, h, logw] = hypergeom_weights(k, N, d);
    keep = logw > max(logw) - abs(log(eps));
    h = h(keep);
    logw = logw(keep);
    mx = max(logw);
    logw = logw - (mx + log(sum(exp(logw - mx))));

    pk = k / N;
    fpc = d * (N - d) / (N - 1);
    delta_mom = sqrt(az / max(fpc * pk * (1 - pk), realmin));
    vI_mom = 1 / (1 + pk * delta_mom / max(1 - pk * delta_mom, 0.15));
    u_mom = log([vI_mom, (1 - vI_mom) / pk]);
    fun = @(u) deepmimo_obj(u, z, R, d, h, logw, lo, hi);
    if ~isempty(u0)
        u_warm = min(max(u0, lo + 1), hi - 1);
        if fun(u_warm) <= fun(u_mom)
            startU = u_warm;
        else
            startU = u_mom;
        end
    else
        startU = u_mom;
    end
    [bestU, best] = fminsearch(fun, startU, opt);
    if ~(all(bestU >= lo) && all(bestU <= hi))
        best = inf;
    end
end

function nll = deepmimo_obj(u, z, R, d, h, logw, lo, hi)
    if any(~isfinite(u)) || any(u < lo) || any(u > hi)
        nll = realmax;
        return;
    end
    V = d * exp(u(1)) + h * exp(u(2));
    A = (logw - 0.5 * R * log(2 * pi * V)) - (0.5 * R * z) * (1 ./ V);
    m = max(A, [], 2);
    nll = -sum(m + log(sum(exp(A - m), 2)));
    if ~isfinite(nll), nll = realmax; end
end
