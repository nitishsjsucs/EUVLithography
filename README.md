# EUVLithography

A copy of the **EUVlitho** research code by Hiroyoshi Tanabe and colleagues
([takahashi-edalab/EUVlitho](https://github.com/takahashi-edalab/EUVlitho), MIT), with a
Streamlit viewer and a matplotlib analysis script added on top for exploring the
simulator's precomputed output.

The electromagnetic simulator and the CNN are the reference implementation for a series
of published papers (listed under [Attribution](#attribution)). Layered on top are
`streamlit_app.py`, `analyze_data.py`, `setup_windows.py` and the Windows launcher — a
visualisation layer over the CSV files the upstream pipeline produces. The original upstream documentation is preserved verbatim
at [`include/README.md`](include/README.md) and is the authoritative guide to the
simulator itself.

## What the upstream code does

EUV lithography masks are not flat. At 13.5 nm with a 60 nm-thick absorber, the mask
behaves as a 3D scatterer, and the thin-mask (Kirchhoff) approximation — just Fourier
transforming the layout — is visibly wrong. Modelling it properly means solving Maxwell's
equations over the mask, which is slow.

The upstream pipeline has two halves:

1. **Rigorous EM simulator** (`emint/intensity.cpp`, `include/header.h`) — a 3D waveguide
   model, equivalent to RCWA but with fewer field components. The committed configuration
   is λ = 13.5 nm, NA = 0.33, 4× magnification, 6° chief ray angle, dipole illumination
   (σ 0.55–0.9, 90° opening), 60 nm absorber (n = 0.9567 + 0.0343i) over a 40-pair
   multilayer, on a 2048 nm pitch sampled at 512×512. It builds on Eigen + oneAPI MKL for
   the dense linear algebra and MAGMA/CUDA for the GPU eigensolves.
2. **CNN surrogate** (`cnn/`, `cnnpredict/`) — six PyTorch Lightning CNNs that predict the
   real and imaginary parts of the M3D parameters a0, ax, ay directly from the mask
   bitmap, skipping the EM solve. Each is five conv/BatchNorm/max-pool blocks
   (16→32→64→128→256 channels, 3×3, **circular padding** so the periodic mask wraps
   correctly) into a 4096-unit FC layer, regressing 1901 diffraction orders for a0 and
   1749 for ax/ay under MSE loss. Training uses pattern-shift augmentation: a random
   cyclic roll of the mask with the matching phase ramp applied analytically to the
   targets, which expands 20,000 rigorous simulations into 1,000,000 training examples.
   Predicted M3D parameters are then fed to either Abbe summation (`cnnabbe/`) or the
   STCC/SOCS formulation (`cnnsocs/`) to form the aerial image.

## Accuracy of the committed example

The repository ships one worked example — the same 512×512 mask run through every path.
Comparing those CSVs directly (parse each file past its 4 header lines, take the array,
subtract elementwise):

| Aerial image | vs. rigorous EM | RMSE | Max abs. error |
|---|---|---|---|
| `cnnabbe/nnabbe.csv` — CNN + Abbe | `emint/emint.csv` | **0.0061** | 0.0242 |
| `cnnsocs/nnsocs.csv` — CNN + SOCS | `emint/emint.csv` | **0.0059** | 0.0236 |
| `cnnabbe/ftint.csv` — thin-mask FT | `emint/emint.csv` | 0.0160 | 0.0392 |

Peak intensity in the EM reference is 0.657, so the CNN's RMSE is about 0.9% of peak, and
roughly 2.6× closer to the rigorous result than the thin-mask approximation. That is the
point of the method, and it is the one quantitative claim this repository can actually
support — it comes from files that are committed here. It is a **single pattern**, not a
benchmark over a test set.

## What is missing

- **No trained weights.** No `model.ckpt` anywhere; the CNNs cannot be run.
- **No training data.** `cnn/data/ampdata/*.npy` and `maskdata/mask.npy` each hold a
  single sample; `cnn/model/*/`*.py* expect `ndata=20000` / `nval=5000` from absolute
  paths under `/home/tanabe/...`, which are not in the repo.
- The C++ makefiles (`emint/makeint` and friends) also carry the original author's
  hardcoded `/home/tanabe/magma` and Eigen paths and need editing before they build.

So the EM simulator and the CNN cannot be reproduced from this checkout alone. What runs
is the visualisation of the already-computed CSVs.

## Running the viewer

```bash
pip install -r streamlit_requirements.txt
streamlit run streamlit_app.py      # or: run_streamlit.bat on Windows
python analyze_data.py              # writes output_mask.png, output_intensity.png, output_comparison.png
```

Both read `emint/mask.csv` and `emint/emint.csv` and plot them. Note that the "Run CNN
Inference" button in the demo tab is a placeholder — it sleeps briefly and prints a fixed
message; no model is loaded.

## Attribution

All C++ and CNN source here is from **[takahashi-edalab/EUVlitho](https://github.com/takahashi-edalab/EUVlitho)**,
MIT License, Copyright (c) 2024 Hiroyoshi Tanabe. The method is described in:

- H. Tanabe, M. Shimode and A. Takahashi, "Rigorous electromagnetic simulator for extreme
  ultraviolet lithography and convolutional neural network reproducing electromagnetic
  simulations," *JM3* **24** (2025) 024201. https://doi.org/10.1117/1.JMM.24.2.024201
- H. Tanabe, A. Jinguji and A. Takahashi, "Weakly guiding approximation of a three
  dimensional waveguide model for extreme ultraviolet lithography simulation," *JOSA A*
  **41** (2024) 1491. https://doi.org/10.1364/JOSAA.516610
- H. Tanabe, S. Sato and A. Takahashi, "Fast EUV lithography simulation using convolutional
  neural network," *JM3* **20** (2021) 041202. https://doi.org/10.1117/1.JMM.20.4.041202
- H. Tanabe and A. Takahashi, "Data augmentation in extreme ultraviolet lithography
  simulation using convolutional neural network," *JM3* **21** (2022) 041602.
  https://doi.org/10.1117/1.JMM.21.4.041602
- H. Tanabe, A. Jinguji and A. Takahashi, "Evaluation of convolutional neural network for
  fast extreme ultraviolet lithography simulation using 3nm node mask patterns," *JM3*
  **22** (2023) 024201. https://doi.org/10.1117/1.JMM.22.2.024201
- H. Tanabe, A. Jinguji and A. Takahashi, "Accelerating extreme ultraviolet lithography
  simulation with weakly guiding approximation and source position dependent transmission
  cross coefficient formula," *JM3* **23** (2024) 014201. https://doi.org/10.1117/1.JMM.23.1.014201

The illumination model in `ampS` (`include/header.h`) follows N. Davydova et al., SPIE 88860A.
