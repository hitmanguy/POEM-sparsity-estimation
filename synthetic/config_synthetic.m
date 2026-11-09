function cfg = config_synthetic()
% CONFIG_SYNTHETIC Central configuration for synthetic POEM benchmarks.
%   Defines dimensions, signal parameters, sensing parameters, and RNG seeds
%   for static (Section V) and sparsity-transition tracking (Table II) runs.
%
% Inputs:
%   None
%
% Outputs:
%   cfg - Struct with fields .static and .tracking (1x1 struct)

    % Static multi-round benchmark (N = 2500, k = 250, M = 250, d = 10)
    cfg.static.N = 2500;
    cfg.static.k = 250;
    cfg.static.M = 250;
    cfg.static.d = 10;
    cfg.static.Es = 0.95;
    cfg.static.muS = 0;
    cfg.static.varS = 400;
    cfg.static.varI = (1 - cfg.static.Es) / cfg.static.Es * ...
        cfg.static.k * (cfg.static.varS + cfg.static.muS^2) / (cfg.static.N - cfg.static.k);
    cfg.static.snr_list = [10, 20];
    cfg.static.R_list = [5, 25, 125];
    cfg.static.trials = 100;
    cfg.static.seed_base = 100000;
    cfg.static.seed_snr_step = 10000;
    cfg.static.sync_paper_rng_stream = true;

    % Sparsity-transition tracking benchmark (Table II: N = 5000, M = 2500)
    cfg.tracking.N = 5000;
    cfg.tracking.M = 2500;
    cfg.tracking.var_s = 5;
    cfg.tracking.var_i = 0.1;
    cfg.tracking.total_rounds = 200;
    cfg.tracking.change_after_round = 100;
    cfg.tracking.checkpoints = [50, 100, 101, 150, 200];
    cfg.tracking.transitions = [200, 300; 400, 500; 100, 600];
    cfg.tracking.snr_list = [Inf, 20, 10, 0];
    cfg.tracking.trials = 40;
    cfg.tracking.alpha_target = 1e-3;
    cfg.tracking.q_min = 0.20;
    cfg.tracking.seed_mult = 10007;
    cfg.tracking.seed_offset = 7919;
end
