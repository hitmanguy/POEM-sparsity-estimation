# Real-World Datasets for POEM Evaluation

Raw datasets are not redistributed in this repository. Follow the steps below to place the required files into `data/raw/` before running the preparation and benchmark scripts.

---

## 1. DeepMIMO Wireless Channel Impulse Responses (Table III)

- **Source:** [DeepMIMO Dataset](https://www.deepmimo.net/) (A. Alkhateeb, *"DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications"*, arXiv:1902.06435, 2019).
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) / DeepMIMO terms of use.
- **Required File:** `cir.mat` containing variable `cir` of size `50 x 1024` (`complex double`, 50 consecutive temporal channel snapshots across $N = 1024$ delay/subcarrier taps).
- **Placement:**
  ```text
  data/raw/cir.mat
  ```
- **Run Pipeline:**
  ```matlab
  startup_poem;
  prepare_deepmimo;
  run_poem_deepmimo;
  ```

---

## 2. EPFL SMLM Challenge 2016 Microscopy Sequence (Table IV)

- **Source:** [EPFL Single-Molecule Localization Microscopy (SMLM) Symposium Challenge 2016](https://srm.epfl.ch/Challenge/ChallengeData) (D. Sage *et al.*, *"Quantitative evaluation of software packages for single-molecule localization microscopy"*, *Nature Methods*, vol. 12, pp. 717–724, 2015).
- **Dataset Sequence:** `MT0.N2.LD` (2D experimental-like high-noise, low-density microtubules sequence).
- **License:** EPFL Biomedical Imaging Group open benchmark terms (cite Sage *et al.*, *Nature Methods* 2015).
- **Exact Frames Used:** Consecutive frames `2391` to `2400` ($R = 1,\dots,10$; $64 \times 64$ pixels, $N = 4096$, $100\text{ nm/pixel}$). Frames `2391`–`2398` contain a stable single active emitter at $(x, y) = (2078, 3809)\text{ nm}$ ($k_{\mathrm{ref}} = 10.1$ pixels), followed by blinking/bleaching in frames `2399`–`2400`.
- **Required Files & Placement:**
  Place the extracted folder `sequence-MT0.N2.LD-2D-Exp-as-stack` inside `data/raw/`:
  ```text
  data/raw/sequence-MT0.N2.LD-2D-Exp-as-stack/
  ├── sequence-as-stack-MT0.N2.LD-2D-Exp.tif
  ├── activations.csv
  └── stats_activation_frames.csv
  ```
- **Run Pipeline:**
  ```matlab
  startup_poem;
  prepare_smlm;
  run_poem_smlm;
  ```
