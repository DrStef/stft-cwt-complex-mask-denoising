# Advanced Speech Denoising with Complex Masks in STFT and CWT

**Dr. Stéphane Dedieu** 
<br>Applied Mathematics | Digital Signal Processing | ML  <br>
January - March 2026  <br>
<a href="https://www.linkedin.com/in/sdedieu/">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" width="30" height="30">
</a>



**Project status:** Early development – first STFT-based U-Net training running right now 😄

## Overview

This project explores **speech denoising** in the time-frequency domain using **complex masking** techniques.  

The main goal is to separate clean speech from realistic background noise using learned or estimated masks applied to noisy spectrograms.  

We start with STFT-based masking (inspired by Clearformer-style approaches) and plan to extend to **Continuous Wavelet Transform (CWT)** for better handling of non-stationary / impulsive noises (helicopter, chainsaw, baby cry, industrial sounds like bearings/drills).

### Key datasets
- **Clean speech**: LibriSpeech dev-clean (selected clean speakers)
- **Noise**: Subset of ESC-10 (rain, sea waves, helicopter, chainsaw, firecrackling, clocktick – pseudo-stationary focus)

### Current pipeline (STFT phase)
- Frame extraction: 1.5 s @ 16 kHz, RMS filtering to remove silences
- Peak normalization per frame → max ±1
- Mixture generation with controlled SNR (focus on +6 dB for first training)
- Anti-clipping: s/2 + n_scaled/2 before addition
- STFT: n_fft=512, hop=94 → [257, 256] spectrograms
- Input: log1p(magnitude) normalized globally (mean=0.1009, std=0.1856)
- Target: complex mask (real + imag), clamped to [-5, 5]
- Model: simple U-Net predicting mask from magnitude spectrogram
- Loss: L1 on mask + waveform reconstruction (to kill musical noise)
- Evaluation: SNR gain, visual spectrograms, perceptual listening

### Why complex masks?
Magnitude-only masking creates musical noise artifacts.  
Complex masks (magnitude + phase correction) reduce these artifacts significantly.


## Model Choice: Simple U-Net vs. Full U-Net

### Overview
We currently use a **Simple U-Net** architecture as the baseline for complex mask estimation in the STFT domain. This lightweight model serves as a first approximation to quickly validate the overall pipeline (data preparation, mixture generation, normalization, complex masking, and waveform reconstruction).

The Simple U-Net is intentionally kept minimal:
- 3 downsampling / upsampling levels
- Filter progression: 32 → 64 → 128 → 256 (bottleneck)
- Basic skip connections without residuals or attention mechanisms
- Total parameters: ~1–2 million (fast training, low memory footprint)

This design allows rapid iteration, debugging of the denoising pipeline, and early perceptual evaluation of denoised audio samples.

### Why Start with Simple U-Net?
- **Speed and simplicity** — Training epochs are short (~10 min each), enabling fast experimentation with mixtures, SNR levels, normalization, clamping bounds, and loss functions (mask + waveform).
- **Lower risk of overfitting** — With only 2000 training frames (1.5 s each at +6 dB), a small model is less likely to memorize specific speakers or phrases.
- **Easier debugging** — Fewer layers mean fewer opportunities for shape mismatches, gradient issues, or vanishing/exploding gradients.
- **Proof of concept** — If the Simple U-Net already produces intelligible speech with reduced noise and minimal musical artifacts, it confirms the core idea (complex masking + waveform loss) is sound.

### When / Why Upgrade to a Full U-Net?
A **Full U-Net** (or enhanced variant) would bring significantly more capacity and performance, especially for challenging cases (lower SNR, impulsive noises like baby cry / helicopter / chainsaw, or industrial transients).

Expected improvements with a Full U-Net:
- Deeper architecture (4–5 levels, filters up to 512–1024)
- Residual connections → better gradient flow and feature reuse
- Attention gates → focus on relevant time-frequency regions (speech formants vs. noise)
- Higher parameter count (~10–30 million) → better modeling of subtle phase corrections and high-frequency details
- Reduced musical noise and phase artifacts (metallic ringing, bottle clinking, watery shimmer)
- Improved speech intelligibility and naturalness, even at moderate-to-low SNR

Typical gains observed in literature (DCCRN, MetricGAN-U, FullSubNet, etc.):
- +2 to +5 dB additional SNR improvement
- +0.2 to +0.5 in PESQ/STOI scores
- Much cleaner perceptual quality (fewer audible artifacts)

### Current Strategy
We begin with the **Simple U-Net** to:
- Validate the entire end-to-end pipeline
- Achieve a working baseline with audible speech enhancement at +6 dB
- Identify remaining limitations (e.g., residual musical noise, weak transients)

Once the Simple U-Net produces clean, intelligible denoised speech with minimal artifacts, we plan to upgrade to a **Full U-Net** (or Attention/Res U-Net variant) to push performance further — especially when scaling to multiple SNR levels, impulsive noises, and eventual CWT-based processing.

This staged approach (simple → full) balances speed of experimentation with final quality.






### Future directions
- Train on multiple SNR levels (0 dB, +12 dB, eventually negative)
- Switch to CWT (complex Morlet wavelet) for better time-frequency resolution on transients
- Target industrial applications: bearing fault detection, drill noise separation in exploration
- Add objective metrics: PESQ, STOI (once training stabilizes)

## Current training status (as of March 2026)
- Running on SNR = +6 dB mixtures (speech audible but noise clearly present)
- U-Net with waveform loss activated
- First epochs in progress – stay tuned for denoised audio samples !

### Acknowledgments
- LibriSpeech: clean speech corpus
- ESC-10: environmental noise dataset
- Clearformer & DCCRN papers: inspiration for complex masking

### Structure of the Repository

complex-mask-denoising/                    # nom de repo GitHub privé (ou denoising-complex-mask, speech-noise-separation, etc.)
├── src/                                   # code source réutilisable (modules Python)
│   ├── __init__.py
│   ├── dataset.py                         # DenoisingDataset class + data loading utils
│   ├── model.py                           # SimpleUNet class + variantes futures (CWT U-Net, etc.)
│   ├── transforms.py                      # STFT / iSTFT / CWT wrappers (torchaudio + pywt/scipy)
│   ├── mixture.py                         # fonctions de génération mixtures (avec SNR control, peak norm, anti-clip)
│   ├── train.py                           # training loop, loss functions (mask + waveform), logger
│   ├── evaluate.py                        # SNR gain, PESQ/STOI, spectro plot, audio save/listen
│   └── utils.py                           # helpers (save/load .npy, random seed, logging, etc.)
├── notebooks/                             # notebooks exploratoires / dev (pas de code critique ici)
│   ├── v01_current.ipynb                  # ton notebook actif
│   ├── v00_old_attempt.ipynb              # archive du premier essai
│   ├── exploration/                       # sous-dossiers pour tests rapides
│   │   ├── mixtures_check.ipynb           # écoute/visualisation SNR
│   │   ├── cwt_prototype.ipynb            # premiers tests CWT
│   │   └── math_notes.ipynb               # notes maths, formules, derivations
├── data/                                  # données (ne pas commit les gros fichiers !)
│   ├── raw/                               # liens ou copies minimales LibriSpeech/ESC-10
│   ├── processed/                         # frames .npy (clean_speech_frames, noise_frames)
│   └── mixtures/                          # .npy par SNR (noisy_mixtures_6dB.npy, etc.)
│       └── samples/                       # .wav pour écoute (noisy/denoised/clean par exemple)
├── models/                                # checkpoints et logs
│   ├── checkpoints/                       # .pth (best_unet_mask.pth, last_epoch.pth)
│   └── logs/                              # tensorboard logs, loss curves .csv
├── results/                               # outputs finaux (visuels, audio, metrics)
│   ├── plots/                             # spectrograms, waveforms, loss curves
│   ├── audio/                             # denoised samples .wav + noisy/clean pour comparaison
│   └── metrics/                           # CSV avec SNR gain, PESQ/STOI, etc.
├── configs/                               # paramètres centralisés (YAML ou JSON)
│   └── config.yaml                        # SNR_list, n_fft, hop_length, bounds, lr, epochs...
├── requirements.txt                       # dépendances (torch, torchaudio, numpy, matplotlib, torchinfo, pesq, pystoi)
├── README.md                              # le draft qu’on a commencé + badges + installation
├── .gitignore                             # ignore data/*, models/checkpoints/*, *.pth, __pycache__, etc.
└── LICENSE                                # MIT ou ce que tu préfères



complex-mask-denoising/                     # Nom de la repo privée
├── src/                                    # Code source réutilisable (modules Python)
│   ├── init.py
│   ├── dataset.py                          # DenoisingDataset + data loading
│   ├── model.py                            # SimpleUNet + futures variantes CWT
│   ├── transforms.py                       # STFT/iSTFT + CWT wrappers
│   ├── mixture.py                          # Génération mixtures (SNR control, peak norm, anti-clip)
│   ├── train.py                            # Boucle training, losses (mask + waveform), logger
│   ├── evaluate.py                         # SNR gain, spectro plots, audio save/play, PESQ/STOI
│   └── utils.py                            # Helpers (save/load .npy, random seed, logging)
├── notebooks/                              # Zone d’expérimentation / développement
│   ├── v01_current.ipynb                   # Notebook principal actif
│   ├── v00_old_attempt.ipynb               # Archive du premier essai
│   └── exploration/                        # Tests rapides / prototypes
│       ├── mixtures_check.ipynb            # Visualisation / écoute SNR
│       ├── cwt_prototype.ipynb             # Premiers tests CWT
│       └── math_notes.ipynb                # Notes maths, formules
├── data/                                   # Données (ne pas commiter les gros fichiers !)
│   ├── raw/                                # Liens ou copies minimales LibriSpeech/ESC-10
│   ├── processed/                          # Frames .npy (clean_speech_frames.npy, noise_frames.npy)
│   └── mixtures/                           # Mixtures .npy par SNR
│       └── samples/                        # .wav pour écoute (noisy/denoised/clean)
├── models/                                 # Checkpoints et logs
│   ├── checkpoints/                        # .pth (best_unet_mask.pth, etc.)
│   └── logs/                               # Loss curves .csv, tensorboard si utilisé
├── results/                                # Outputs finaux (visuels, audio, metrics)
│   ├── plots/                              # Spectrograms, waveforms, loss curves
│   ├── audio/                              # Samples denoised .wav + comparaisons
│   └── metrics/                            # CSV SNR gain, PESQ/STOI, etc.
├── configs/                                # Paramètres centralisés
│   └── config.yaml                         # SNR_list, n_fft, hop_length, bounds, lr, etc.
├── requirements.txt                        # Dépendances (torch, torchaudio, numpy, torchinfo, pesq, pystoi)
├── README.md                               # Ce fichier
├── .gitignore                              # Ignore data/, *.pth, *.npy, checkpoints/, pycache/
└── LICENSE                                 # MIT (ou autre)












## Setup & Requirements

```bash
pip install torch torchaudio numpy matplotlib scipy pesq pystoi torchinfo
