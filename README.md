# POEM: Profile Occupancy–Energy Mixture Sparsity Estimator

**POEM** is a MATLAB implementation of a multi-round composite maximum likelihood estimator that jointly infers the sparsity order $k$, active coefficient variance $\sigma_s^2$, and background coefficient variance $\sigma_i^2$ of compressible signals from $d$-regular sparse binary compressive measurements. Each sensing row sums $d$ coordinates drawn uniformly without replacement, inducing an exact hypergeometric distribution over support occupancy and a finite scaled-$\chi^2_R$ mixture over $R$-round pooled row energies. Closed-form moment inversion combined with lower-tail bisection yields an operational search bracket $[k_1, k_2]$, within which the nuisance variances are profiled out via derivative-free Nelder–Mead optimization.

---

## Requirements

- **MATLAB:** R2022b or later (tested on R2025b and R2026a).
- **Required Toolboxes:** None for core estimation (`fminsearch`, `gammainc`, `gammaincinv`, and `gammaln` are built into base MATLAB).
- **Optional Toolboxes:** Parallel Computing Toolbox (`parfor` accelerates the 100-trial Monte Carlo benchmark scripts; falls back to serial execution automatically if unavailable).

---

## Quick Start

From the repository root in MATLAB:

```matlab
startup_poem;   % Add src/, synthetic/, realdata/, examples/, and tests/ to path
run_all_tests;  % Verify Proposition 1 moment inversion and [k1, k2] bracket coverage
demo_poem;      % Run a 25-round synthetic estimation example (N=2500, k=250, M=250, d=10)
```

Or call the top-level estimator directly on an $M \times R$ measurement matrix `Y` (or $M \times 1$ pooled energy vector `s`):

```matlab
[k_hat, sig_s2, sig_i2, info] = poem_estimate(Y, R, N, d, noiseVar);
```

---

## Reproducing the Paper

All benchmark scripts use fixed random seeds, print formatted tables matching the paper, and write `.mat` and `.csv` outputs to `results/`.

| Paper Reference | Description | Script | Typical Runtime (8 workers) |
| :--- | :--- | :--- | :--- |
| **Sec. V (Static Benchmark)** | 100-trial synthetic benchmark ($N=2500, k=250, M=250, d=10$, $\mathrm{SNR}\in\{10,20\}\text{ dB}$, $R\in\{5,25,125\}$): $\hat{k}$, MAPE $\pm 95\%$ CI, bracket $[k_1,k_2]$ coverage | `synthetic/run_poem_static.m` | ~25 s |
| **Table II** | Synthetic sparsity-transition tracking ($200\to 300$, $400\to 500$, $100\to 600$ at round 101; $\mathrm{SNR}\in\{\infty,20,10,0\}\text{ dB}$) | `synthetic/run_poem_tracking.m` | ~3 min |
| **Table III** | DeepMIMO wireless channel evaluation ($N=1024, M=250, d=10, n_{\mathrm{win}}=10$, $R\in\{5,10,20,40,50\}$, 100 trials) | `realdata/deepmimo/run_poem_deepmimo.m` | ~15 s |
| **Table IV** | EPFL SMLM microscopy evaluation (frames 2391–2400, $N=4096, M=2048, d=60$, $R=1\dots 10$, 100 and 20 trials) | `realdata/smlm/run_poem_smlm.m` | ~45 s |

---

## Real-World Data Instructions

Raw datasets for **Table III** (DeepMIMO) and **Table IV** (EPFL SMLM Challenge 2016) are not redistributed in this repository. See [`realdata/README.md`](realdata/README.md) for download links, dataset licenses, exact frame/file specifications, and where to place them under `data/raw/`.

---

## Repository Structure

```text
poem-sparsity-estimation/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── startup_poem.m
├── src/
│   ├── poem_estimate.m
│   ├── core/
│   │   ├── poem_bracket.m
│   │   ├── poem_profile_mle.m
│   │   └── poem_composite_loglik.m
│   ├── sensing/
│   │   ├── poem_random_bsm.m
│   │   └── poem_noise_variance.m
│   └── utils/
│       ├── log_choose.m
│       └── hypergeom_weights.m
├── synthetic/
│   ├── gen_synthetic_signal.m
│   ├── run_poem_static.m
│   ├── run_poem_tracking.m
│   └── config_synthetic.m
├── realdata/
│   ├── README.md
│   ├── deepmimo/
│   │   ├── prepare_deepmimo.m
│   │   └── run_poem_deepmimo.m
│   └── smlm/
│       ├── prepare_smlm.m
│       └── run_poem_smlm.m
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── examples/
│   └── demo_poem.m
└── tests/
    ├── test_bracket_contains_k.m
    ├── test_moment_inversion_exact.m
    └── run_all_tests.m
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{chauhan2027poem,
  author    = {Chauhan, Sahil},
  title     = {Joint Sparsity-Order and Variance Estimation from Sparse Binary Compressive Measurements via Multi-Round Composite Likelihood},
  booktitle = {Proc. IEEE Wireless Communications and Networking Conference (WCNC)},
  year      = {2027},
  note      = {Under review}
}
```

---

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.
