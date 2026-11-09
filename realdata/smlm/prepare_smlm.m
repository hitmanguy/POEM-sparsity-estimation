function data_out = prepare_smlm(raw_smlm_dir)
% PREPARE_SMLM Extract SMLM frames 2391-2400 and ground-truth PSF sparsity from local dataset.
%   Computes 2D orthonormal DCT coefficients X_dct (4096x10) and ground-truth
%   sparsity k_gt = 10.1 pixels from activations.csv (Section V-C).
%
% Inputs:
%   raw_smlm_dir - Optional folder path containing SMLM .tif and .csv files
%                  (defaults to data/raw/sequence-MT0.N2.LD-2D-Exp-as-stack)
%
% Outputs:
%   data_out     - Struct saved to data/processed/smlm_frames_2391_2400.mat

    root_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    if nargin < 1 || isempty(raw_smlm_dir)
        raw_smlm_dir = fullfile(root_dir, 'data', 'raw', 'sequence-MT0.N2.LD-2D-Exp-as-stack');
    end

    tifPath = fullfile(raw_smlm_dir, 'sequence-as-stack-MT0.N2.LD-2D-Exp.tif');
    csvStats = fullfile(raw_smlm_dir, 'stats_activation_frames.csv');
    csvAct = fullfile(raw_smlm_dir, 'activations.csv');

    assert(isfile(tifPath), 'SMLM TIFF stack not found at %s. See realdata/README.md.', tifPath);
    assert(isfile(csvStats), 'SMLM stats CSV not found at %s. See realdata/README.md.', csvStats);
    assert(isfile(csvAct), 'SMLM activations CSV not found at %s. See realdata/README.md.', csvAct);

    warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
    T_stats = readtable(csvStats);
    T_act = readtable(csvAct);

    frame_seq = 2391:2400;
    numRounds = numel(frame_seq);
    R_stable = 8;

    img_sample = double(imread(tifPath, frame_seq(1)));
    [H, W] = size(img_sample);
    N = H * W;

    gt_nbfluos = zeros(numRounds, 1);
    gt_photons = zeros(numRounds, 1);
    gt_k_true = zeros(numRounds, 1);
    X_dct = zeros(N, numRounds);

    for r = 1:numRounds
        f = frame_seq(r);
        im = double(imread(tifPath, f));
        sub = im - median(im(:));

        C = orthonormal_dct2(sub);
        X_dct(:, r) = C(:);

        gt_nbfluos(r) = T_stats.nbfluos(f);
        gt_photons(r) = T_stats.nbphotons(f);

        act_f = T_act(T_act.frame == f, :);
        mask_psf = false(H, W);
        for m = 1:height(act_f)
            col_c = round(act_f.xnano(m) / 100) + 1;
            row_c = round(act_f.ynano(m) / 100) + 1;
            r_rng = max(1, row_c - 2):min(H, row_c + 2);
            c_rng = max(1, col_c - 2):min(W, col_c + 2);
            mask_psf(r_rng, c_rng) = true;
        end

        out_px = sub(~mask_psf);
        bg_std = std(out_px);
        var_bg = mean(out_px.^2);

        if height(act_f) > 0
            in_px = sub(mask_psf);
            active_px = in_px(in_px > 3.5 * bg_std);
            if numel(active_px) >= 2
                sig_sq = max(0, active_px.^2 - var_bg);
                cs2 = var(sig_sq) / (mean(sig_sq)^2);
                gt_k_true(r) = numel(active_px) / (1 + cs2);
            else
                gt_k_true(r) = max(1, numel(active_px));
            end
        else
            gt_k_true(r) = 0;
        end
    end

    k_gt = mean(gt_k_true(1:R_stable));

    proc_dir = fullfile(root_dir, 'data', 'processed');
    if ~isfolder(proc_dir), mkdir(proc_dir); end
    out_file = fullfile(proc_dir, 'smlm_frames_2391_2400.mat');

    data_out = struct( ...
        'X_dct', X_dct, ...
        'frame_seq', frame_seq, ...
        'N', N, ...
        'H', H, ...
        'W', W, ...
        'R_stable', R_stable, ...
        'k_gt', k_gt, ...
        'gt_k_true', gt_k_true, ...
        'gt_nbfluos', gt_nbfluos, ...
        'gt_photons', gt_photons);
    save(out_file, '-struct', 'data_out');
    fprintf('Prepared SMLM frames %d-%d (N = %d, k_gt = %.1f) -> %s\n', ...
        frame_seq(1), frame_seq(end), N, k_gt, out_file);
end

function C = orthonormal_dct2(img)
    [nr, nc] = size(img);
    C = zeros(nr, nc);
    for col = 1:nc, C(:, col) = odct_1d(img(:, col)); end
    for row = 1:nr, C(row, :) = odct_1d(C(row, :).').'; end
end

function c = odct_1d(v)
    v = v(:);
    n_len = numel(v);
    ve = [v; v(end:-1:1)];
    V = fft(ve);
    V = V(1:n_len);
    k_vec = (0:n_len - 1)';
    c = real(V .* exp(-1j * pi * k_vec / (2 * n_len)));
    w = sqrt(2 / n_len) * ones(n_len, 1);
    w(1) = sqrt(1 / n_len);
    c = c .* (0.5 * w);
end
