function results = run_poem_tracking()
% RUN_POEM_TRACKING Reproduce Table II synthetic sparsity-transition tracking results.
%   Evaluates POEM with dual-CUSUM support change detection (Fig. 1, Eqs. 15-19)
%   derived from modified_mle_energy_v2.m across transitions 200->300, 400->500,
%   and 100->600 (change at round 101) for SNR = [Inf, 20, 10, 0] dB.
%
% Inputs:
%   None
%
% Outputs:
%   results - Struct containing per-transition and per-SNR tracking statistics

    t_start = tic;
    root_dir = fileparts(fileparts(mfilename('fullpath')));
    run(fullfile(root_dir, 'startup_poem.m'));

    % --- Configuration Block ---
    cfg = config_synthetic();
    tr = cfg.tracking;
    N = tr.N;
    M = tr.M;
    var_s = tr.var_s;
    var_i = tr.var_i;
    totalRounds = tr.total_rounds;
    changeAfter = tr.change_after_round;
    checkpoints = tr.checkpoints;
    transitions = tr.transitions;
    snr_list = tr.snr_list;
    trials = tr.trials;
    alpha_target = tr.alpha_target;
    q_min = tr.q_min;
    alpha_det = alpha_target / 3;

    nTrans = size(transitions, 1);
    nSnr = numel(snr_list);
    nCp = numel(checkpoints);

    MEAN_K = nan(nSnr, nCp, nTrans);
    STD_K = nan(nSnr, nCp, nTrans);
    RMSE_K = nan(nSnr, nCp, nTrans);
    COV_K = nan(nSnr, nCp, nTrans);

    fprintf('================================================================================================\n');
    fprintf('POEM SPARSITY-TRANSITION TRACKING (Table II: N = %d, M = %d, Trials = %d, Change at Round %d)\n', ...
        N, M, trials, changeAfter + 1);
    fprintf('================================================================================================\n');

    for it = 1:nTrans
        k_init = transitions(it, 1);
        k_final = transitions(it, 2);
        d = max(1, min(N - 1, round(N * (1 - exp(-1.6 / k_init)))));

        kRound = [repmat(k_init, 1, changeAfter), repmat(k_final, 1, totalRounds - changeAfter)];
        segIdx = [ones(1, changeAfter), 2 * ones(1, totalRounds - changeAfter)];

        Pclean_initial = d * (var_i + (k_init / N) * (var_s - var_i));

        for is = 1:nSnr
            snr_dB = snr_list(is);
            if isfinite(snr_dB)
                var_meas = Pclean_initial / (10^(snr_dB / 10));
            else
                var_meas = 0;
            end

            khatAd = nan(trials, nCp);
            k1Ad = nan(trials, nCp);
            k2Ad = nan(trials, nCp);

            parfor t = 1:trials
                rng(tr.seed_mult * t + tr.seed_offset, 'twister');
                B = poem_random_bsm(M, N, d);
                S = [];

                s_total = zeros(M, 1);
                n_total = 0;
                s_cand = zeros(M, 2);
                n_cand = zeros(1, 2);
                S_rho = 0;
                S_lvlP = 0;
                S_lvlN = 0;

                wCntR = 0; wM2R = 0; zCandR = [];
                wCntL = 0; wM2L = 0; zCandL = [];

                kA = nan(1, nCp);
                k1A = nan(1, nCp);
                k2A = nan(1, nCp);

                for r = 1:totalRounds
                    if r == 1 || segIdx(r) ~= segIdx(r - 1)
                        S = randperm(N, kRound(r));
                    end
                    x = gen_synthetic_signal(N, S, 0, var_s, var_i);
                    y_clean = B * x;
                    if isfinite(snr_dB)
                        noise_unit = randn(M, 1);
                        y = y_clean + sqrt(var_meas) * noise_unit;
                    else
                        y = y_clean;
                    end
                    y2 = y.^2;

                    n_base_r = n_total - n_cand(1);
                    n_base_l = n_total - n_cand(2);

                    if n_base_r >= 2 && n_base_l >= 2
                        sc_r = (s_total - s_cand(:, 1)) / n_base_r;
                        mu_r = mean(sc_r);
                        bh_r = mean(sc_r.^2) / (1 + 2 / n_base_r);
                        ah_r = bh_r - mu_r^2;

                        sc_l = (s_total - s_cand(:, 2)) / n_base_l;
                        mu_l = mean(sc_l);
                        bh_l = mean(sc_l.^2) / (1 + 2 / n_base_l);

                        if ah_r > 0 && mu_r > 0 && mu_l > 0
                            rho0 = ah_r / sqrt((ah_r + 2 * bh_r) * (ah_r + 2 * bh_r / n_base_r));
                            u_vec = (y2 - mean(y2)) / std(y2);
                            v_vec = (sc_r - mu_r) / std(sc_r);
                            rho_hat = mean(u_vec .* v_vec);
                            infl = u_vec .* v_vec - 0.5 * rho0 * (u_vec.^2 + v_vec.^2);
                            se_sand = sqrt((var(infl) / M) * (1 + 1 / n_base_r));
                            zRawR = (rho_hat - rho0) / se_sand;

                            lvl = mean(y2) / mu_l;
                            se_lvl1 = sqrt((2 * bh_l / (mu_l^2 * M)) * (1 + 1 / n_base_l));
                            zRawL = (lvl - 1) / se_lvl1;

                            psiRho = psi_ucb(wM2R, wCntR, alpha_det);
                            psiLvl = psi_ucb(wM2L, wCntL, alpha_det);

                            armedR = false; armedL = false;
                            h_rho = inf; h_lvl = inf;
                            if isfinite(psiRho) && se_sand > 0
                                se_rho = sqrt(psiRho) * se_sand;
                                delta_rho = rho0 * q_min / se_rho;
                                kappa_rho = 0.5 * delta_rho;
                                if kappa_rho > 0
                                    h_rho = solve_siegmund_h(kappa_rho, alpha_det);
                                    armedR = ((h_rho + 1.166) / kappa_rho <= n_base_r);
                                end
                            else
                                kappa_rho = 0;
                            end

                            if isfinite(psiLvl) && se_lvl1 > 0
                                se_lvl = sqrt(psiLvl) * se_lvl1;
                                delta_lvl = q_min / se_lvl;
                                kappa_lvl = 0.5 * delta_lvl;
                                if kappa_lvl > 0
                                    h_lvl = solve_siegmund_h(kappa_lvl, alpha_det);
                                    armedL = ((h_lvl + 1.166) / kappa_lvl <= n_base_l);
                                end
                            else
                                kappa_lvl = 0;
                            end

                            if armedR
                                z_rho = zRawR / sqrt(psiRho);
                                S_rho = max(0, S_rho - z_rho - kappa_rho);
                            else
                                S_rho = 0;
                            end

                            if armedL
                                z_lvl = zRawL / sqrt(psiLvl);
                                S_lvlP = max(0, S_lvlP + z_lvl - kappa_lvl);
                                S_lvlN = max(0, S_lvlN - z_lvl - kappa_lvl);
                            else
                                S_lvlP = 0;
                                S_lvlN = 0;
                            end

                            s_total = s_total + y2;
                            n_total = n_total + 1;

                            if S_rho > 0
                                s_cand(:, 1) = s_cand(:, 1) + y2;
                                n_cand(1) = n_cand(1) + 1;
                                zCandR(end + 1) = zRawR; %#ok<AGROW>
                            else
                                s_cand(:, 1) = 0;
                                n_cand(1) = 0;
                                allZR = [zCandR, zRawR];
                                zCandR = [];
                                wCntR = wCntR + numel(allZR);
                                wM2R = wM2R + sum(allZR.^2);
                            end

                            if S_lvlP > 0 || S_lvlN > 0
                                s_cand(:, 2) = s_cand(:, 2) + y2;
                                n_cand(2) = n_cand(2) + 1;
                                zCandL(end + 1) = zRawL; %#ok<AGROW>
                            else
                                s_cand(:, 2) = 0;
                                n_cand(2) = 0;
                                allZL = [zCandL, zRawL];
                                zCandL = [];
                                wCntL = wCntL + numel(allZL);
                                wM2L = wM2L + sum(allZL.^2);
                            end

                            alarmR = armedR && (S_rho >= h_rho);
                            alarmL = armedL && (S_lvlP >= h_lvl || S_lvlN >= h_lvl);
                            if alarmR || alarmL
                                if alarmL && (~alarmR || n_cand(2) <= n_cand(1))
                                    s_total = s_cand(:, 2);
                                    n_total = n_cand(2);
                                else
                                    s_total = s_cand(:, 1);
                                    n_total = n_cand(1);
                                end
                                s_cand(:) = 0;
                                n_cand(:) = 0;
                                S_rho = 0; S_lvlP = 0; S_lvlN = 0;
                                zCandR = []; zCandL = [];
                            end
                        else
                            s_total = s_total + y2;
                            n_total = n_total + 1;
                        end
                    else
                        s_total = s_total + y2;
                        n_total = n_total + 1;
                    end

                    ci = find(checkpoints == r, 1);
                    if ~isempty(ci)
                        s_ad = s_total / max(n_total, 1);
                        [bLo, bHi] = bounds_k1k2_tracking(s_ad, n_total, N, d, isfinite(snr_dB));
                        kA(ci) = estimate_g_tracking(s_ad, n_total, N, d, bLo, bHi);
                        k1A(ci) = bLo;
                        k2A(ci) = bHi;
                    end
                end

                khatAd(t, :) = kA;
                k1Ad(t, :) = k1A;
                k2Ad(t, :) = k2A;
            end

            for ci = 1:nCp
                kTrue = kRound(checkpoints(ci));
                kh = khatAd(:, ci);
                err = kh - kTrue;
                MEAN_K(is, ci, it) = mean(kh, 'omitnan');
                STD_K(is, ci, it) = std(kh, 'omitnan');
                RMSE_K(is, ci, it) = sqrt(mean(err.^2, 'omitnan'));
                COV_K(is, ci, it) = 100 * mean(k1Ad(:, ci) <= kTrue & kTrue <= k2Ad(:, ci));
            end
        end
    end

    % Print Table II format
    fprintf('\n%-8s | %-5s | %-25s | %-25s | %-25s\n', ...
        'SNR (dB)', 'Round', '200->300 (Mean+-SD, RMSE)', '400->500 (Mean+-SD, RMSE)', '100->600 (Mean+-SD, RMSE)');
    fprintf('------------------------------------------------------------------------------------------------\n');
    for is = 1:nSnr
        if isinf(snr_list(is)), snr_str = 'Inf'; else, snr_str = sprintf('%d', snr_list(is)); end
        for ci = 1:nCp
            c1 = sprintf('%6.1f+-%6.2f (%6.2f)', MEAN_K(is, ci, 1), STD_K(is, ci, 1), RMSE_K(is, ci, 1));
            c2 = sprintf('%6.1f+-%6.2f (%6.2f)', MEAN_K(is, ci, 2), STD_K(is, ci, 2), RMSE_K(is, ci, 2));
            c3 = sprintf('%6.1f+-%6.2f (%6.2f)', MEAN_K(is, ci, 3), STD_K(is, ci, 3), RMSE_K(is, ci, 3));
            fprintf('%-8s | %-5d | %-25s | %-25s | %-25s\n', snr_str, checkpoints(ci), c1, c2, c3);
        end
        fprintf('------------------------------------------------------------------------------------------------\n');
    end

    elapsed_sec = toc(t_start);
    out_dir = fullfile(root_dir, 'results');
    if ~isfolder(out_dir), mkdir(out_dir); end

    results = struct( ...
        'MEAN_K', MEAN_K, ...
        'STD_K', STD_K, ...
        'RMSE_K', RMSE_K, ...
        'COV_K', COV_K, ...
        'config', tr, ...
        'matlab_version', version, ...
        'elapsed_sec', elapsed_sec);

    out_mat = fullfile(out_dir, 'poem_tracking_table2.mat');
    save(out_mat, '-struct', 'results');
    fprintf('Saved results to %s\n', out_mat);
    fprintf('MATLAB version: %s | Elapsed time: %.2f s\n', version, elapsed_sec);
end

function psi = psi_ucb(wM2, wCnt, alpha_ucb)
    nu = wCnt - 1;
    if nu < 2
        psi = inf;
        return;
    end
    chi2_lo = 2 * gammaincinv(alpha_ucb, nu / 2);
    psi = max(1.0, wM2 / max(chi2_lo, realmin));
end

function h = solve_siegmund_h(kappa, alpha)
    target = 1 / alpha;
    f = @(hv) (exp(2 * kappa * (hv + 1.166)) - 2 * kappa * (hv + 1.166) - 1) / (2 * kappa^2) - target;
    h_lo = 0;
    h_hi = 20;
    while f(h_hi) < 0 && h_hi < 1e4
        h_hi = h_hi * 2;
    end
    for it = 1:60
        h_mid = 0.5 * (h_lo + h_hi);
        if f(h_mid) > 0, h_hi = h_mid; else, h_lo = h_mid; end
        if abs(h_hi - h_lo) < 1e-4, break; end
    end
    h = 0.5 * (h_lo + h_hi);
end

function [k1, k2] = bounds_k1k2_tracking(s, R, N, d, is_noisy)
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
    if is_noisy
        q_use = q_tail;
    else
        q_use = min(0.05, q_tail + 1.96 * sqrt(q_tail * (1 - q_tail) / M_samples));
    end
    g_q = (2 / R) * gammaincinv(q_tail, R / 2);
    sigma_i_est2 = (quantile(s, q_use) / g_q) / d;

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
    k2 = k_hi;
end

function kHat = estimate_g_tracking(s, R, N, d, kLo, kHi)
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
            [nll(j), u] = fit_at_k_tracking(grid(j), z, az, R, N, d, u, lo, hi, opt);
            U(j, :) = u;
        end
        [~, jb] = min(nll);
        a = grid(max(1, jb - 1));
        b = grid(min(numel(grid), jb + 1));
        u = U(jb, :);
    end
    grid = a:b;
    nll = inf(size(grid));
    for j = 1:numel(grid)
        [nll(j), u] = fit_at_k_tracking(grid(j), z, az, R, N, d, u, lo, hi, opt);
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
end

function [best, bestU] = fit_at_k_tracking(k, z, az, R, N, d, u0, lo, hi, opt)
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
    fun = @(u) tracking_obj(u, z, R, d, h, logw, lo, hi);
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

function nll = tracking_obj(u, z, R, d, h, logw, lo, hi)
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
