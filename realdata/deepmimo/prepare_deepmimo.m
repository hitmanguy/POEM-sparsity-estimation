function data_out = prepare_deepmimo(raw_mat_path)
% PREPARE_DEEPMIMO Build normalized complex channel snapshots from local DeepMIMO cir.mat.
%   Computes globally normalized complex channel matrix X (50x1024) and per-snapshot
%   normalized power profiles P for 95%-energy reference k_0.95 (Section V-B, Table III).
%
% Inputs:
%   raw_mat_path - Optional path to cir.mat (defaults to data/raw/cir.mat)
%
% Outputs:
%   data_out     - Struct saved to data/processed/deepmimo_snapshots.mat

    root_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    if nargin < 1 || isempty(raw_mat_path)
        raw_mat_path = fullfile(root_dir, 'data', 'raw', 'cir.mat');
    end
    assert(isfile(raw_mat_path), ...
        'DeepMIMO file not found at %s. See realdata/README.md for instructions.', raw_mat_path);

    S_in = load(raw_mat_path);
    assert(isfield(S_in, 'cir'), 'Expected variable ''cir'' inside %s.', raw_mat_path);
    cir = S_in.cir;

    X = double(cir) / sqrt(mean(abs(cir(:)).^2));
    [totalRounds, N] = size(X);
    P = abs(X ./ sqrt(sum(abs(X).^2, 2))).^2;
    k95f = @(p) find(cumsum(sort(p, 'descend')) / sum(p) >= 0.95, 1);
    kLink = arrayfun(@(j) k95f(P(j, :)), 1:size(P, 1));

    proc_dir = fullfile(root_dir, 'data', 'processed');
    if ~isfolder(proc_dir), mkdir(proc_dir); end
    out_file = fullfile(proc_dir, 'deepmimo_snapshots.mat');

    data_out = struct( ...
        'X', X, ...
        'P', P, ...
        'kLink', kLink, ...
        'N', N, ...
        'totalRounds', totalRounds);
    save(out_file, '-struct', 'data_out');
    fprintf('Prepared DeepMIMO snapshots (%d rounds x %d taps) -> %s\n', totalRounds, N, out_file);
end
