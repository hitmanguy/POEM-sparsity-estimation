function results = run_poem_static(sync_rng_override)
% RUN_POEM_STATIC Reproduce static synthetic POEM results (Section V).
%   Evaluates POEM under the fixed-support two-variance Gaussian model
%   (Eqs. 7-19) for N = 2500, k = 250, M = 250, d = 10, SNR = [10, 20] dB,
%   and R = [5, 25, 125] over 100 Monte Carlo trials.
%
% Inputs:
%   sync_rng_override - Optional logical scalar. If true (default), steps
%                       the per-trial RNG state identically to the paper's
%                       shared-stream experiment. If false, uses a standalone
%                       POEM-only RNG stream.
%
% Outputs:
%   results           - Struct with KHAT, KLO, KHI, KBOUNDARY, summary metrics

    t_start = tic;
    root_dir = fileparts(fileparts(mfilename('fullpath')));
    run(fullfile(root_dir, 'startup_poem.m'));

    % --- Configuration Block ---
    cfg = config_synthetic();
    st = cfg.static;
    if nargin >= 1 && ~isempty(sync_rng_override)
        st.sync_paper_rng_stream = logical(sync_rng_override);
    end

    N = st.N;
    kTrue = st.k;
    M = st.M;
    d = st.d;
    muS = st.muS;
    varS = st.varS;
    varI = st.varI;
    snr_list = st.snr_list;
    R_list = st.R_list;
    trials = st.trials;
    Rmax = max(R_list);
    nS = numel(snr_list);
    nR = numel(R_list);

    KHAT = nan(nS, trials, nR);
    SIG_S2 = nan(nS, trials, nR);
    SIG_I2 = nan(nS, trials, nR);
    KLO = nan(nS, trials, nR);
    KHI = nan(nS, trials, nR);
    KBOUNDARY = false(nS, trials, nR);
    NAN_FAIL = false(nS, trials, nR);
    INVERT_FAIL = false(nS, trials, nR);

    fprintf('========================================================================================\n');
    fprintf('POEM STATIC SYNTHETIC BENCHMARK (N = %d, k = %d, M = %d, d = %d, Trials = %d)\n', ...
        N, kTrue, M, d, trials);
    fprintf('========================================================================================\n');

    for is = 1:nS
        snr = snr_list(is);
        noiseVar = poem_noise_variance(N, kTrue, d, muS, varS, varI, snr);

        khat_s = nan(trials, nR);
        sigs_s = nan(trials, nR);
        sigi_s = nan(trials, nR);
        klo_s = nan(trials, nR);
        khi_s = nan(trials, nR);
        bnd_s = false(trials, nR);
        nan_s = false(trials, nR);
        inv_s = false(trials, nR);

        parfor trial = 1:trials
            seed = st.seed_base + st.seed_snr_step * is + trial;
            rng(seed, 'twister');

            S = randperm(N, kTrue);
            B = poem_random_bsm(M, N, d);

            if st.sync_paper_rng_stream
                advance_trial_rng(M, N);
            end

            e_poem = zeros(M, 1);
            loc_k = nan(1, nR);
            loc_vs = nan(1, nR);
            loc_vi = nan(1, nR);
            loc_lo = nan(1, nR);
            loc_hi = nan(1, nR);
            loc_bnd = false(1, nR);
            loc_nan = false(1, nR);
            loc_inv = false(1, nR);

            for r = 1:Rmax
                x = gen_synthetic_signal(N, S, muS, varS, varI);
                y = B * x + sqrt(noiseVar) * randn(M, 1);
                e_poem = e_poem + y.^2;

                if st.sync_paper_rng_stream
                    advance_round_rng(M, N);
                end

                cp = find(R_list == r, 1);
                if ~isempty(cp)
                    sm = e_poem / r;
                    [kh, vs, vi, info] = poem_estimate(sm, r, N, d, noiseVar);
                    loc_k(cp) = kh;
                    loc_vs(cp) = vs;
                    loc_vi(cp) = vi;
                    loc_lo(cp) = info.k1;
                    loc_hi(cp) = info.k2;
                    loc_bnd(cp) = info.boundary_hit;
                    loc_nan(cp) = info.nan_fail;
                    loc_inv(cp) = info.invert_fail;
                end
            end

            khat_s(trial, :) = loc_k;
            sigs_s(trial, :) = loc_vs;
            sigi_s(trial, :) = loc_vi;
            klo_s(trial, :) = loc_lo;
            khi_s(trial, :) = loc_hi;
            bnd_s(trial, :) = loc_bnd;
            nan_s(trial, :) = loc_nan;
            inv_s(trial, :) = loc_inv;
        end

        KHAT(is, :, :) = khat_s;
        SIG_S2(is, :, :) = sigs_s;
        SIG_I2(is, :, :) = sigi_s;
        KLO(is, :, :) = klo_s;
        KHI(is, :, :) = khi_s;
        KBOUNDARY(is, :, :) = bnd_s;
        NAN_FAIL(is, :, :) = nan_s;
        INVERT_FAIL(is, :, :) = inv_s;
    end

    fprintf('%-7s | %-5s | %-10s | %-18s | %-18s | %-12s | %-12s\n', ...
        'SNR', 'R', 'Mean k_hat', 'MAPE (95% CI)', 'Mean [k1, k2]', 'Coverage (%)', 'Boundary (%)');
    fprintf('----------------------------------------------------------------------------------------\n');

    summary_rows = [];
    for is = 1:nS
        snr = snr_list(is);
        for ir = 1:nR
            R = R_list(ir);
            kh = squeeze(KHAT(is, :, ir));
            lo = squeeze(KLO(is, :, ir));
            hi = squeeze(KHI(is, :, ir));
            bnd = squeeze(KBOUNDARY(is, :, ir));

            valid = isfinite(kh);
            err = abs(kh(valid) - kTrue) / kTrue;
            mean_k = mean(kh(valid));
            mape = 100 * mean(err);
            ci = 100 * 1.96 * std(err) / sqrt(sum(valid));

            valid_b = isfinite(lo) & isfinite(hi) & (lo <= hi);
            mean_lo = mean(lo(valid_b));
            mean_hi = mean(hi(valid_b));
            cov_pct = 100 * mean(valid_b & (lo <= kTrue) & (kTrue <= hi));
            bnd_pct = 100 * mean(bnd);

            fprintf('%-4d dB | %-5d | %10.2f | %5.2f%% [+-%4.2f%%]  | [%6.1f, %6.1f]  | %11.0f%% | %11.0f%%\n', ...
                snr, R, mean_k, mape, ci, mean_lo, mean_hi, cov_pct, bnd_pct);

            summary_rows = [summary_rows; snr, R, mean_k, mape, ci, mean_lo, mean_hi, cov_pct, bnd_pct]; %#ok<AGROW>
        end
        fprintf('----------------------------------------------------------------------------------------\n');
    end

    elapsed_sec = toc(t_start);
    out_dir = fullfile(root_dir, 'results');
    if ~isfolder(out_dir), mkdir(out_dir); end

    results = struct( ...
        'KHAT', KHAT, ...
        'SIG_S2', SIG_S2, ...
        'SIG_I2', SIG_I2, ...
        'KLO', KLO, ...
        'KHI', KHI, ...
        'KBOUNDARY', KBOUNDARY, ...
        'NAN_FAIL', NAN_FAIL, ...
        'INVERT_FAIL', INVERT_FAIL, ...
        'summary', summary_rows, ...
        'config', st, ...
        'matlab_version', version, ...
        'elapsed_sec', elapsed_sec);

    out_mat = fullfile(out_dir, 'poem_static_100trials.mat');
    save(out_mat, '-struct', 'results');

    T = array2table(summary_rows, 'VariableNames', ...
        {'SNR_dB', 'R', 'Mean_k_hat', 'MAPE_pct', 'CI95_pct', 'Mean_k1', 'Mean_k2', 'Coverage_pct', 'Boundary_pct'});
    out_csv = fullfile(out_dir, 'poem_static_100trials.csv');
    writetable(T, out_csv);

    fprintf('Saved results to %s\n', out_mat);
    fprintf('MATLAB version: %s | Elapsed time: %.2f s\n', version, elapsed_sec);
end

function advance_trial_rng(M, N)
    m1 = rand(M, N) < (1 / 1200);
    randn(nnz(m1), 1);
    m2 = rand(M, N) < (1 / 400);
    randn(nnz(m2), 1);
end

function advance_round_rng(M, N)
    randn(M, 1);
    randn(M, 1);
    Mc = floor(M / 2);
    Mg = M - Mc;
    rand(Mc, N);
    randn(Mg, N);
end
