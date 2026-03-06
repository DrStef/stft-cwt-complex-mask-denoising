# Advanced Speech Denoising with Complex Masks in STFT and CWT

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

### Future directions
- Train on multiple SNR levels (0 dB, +12 dB, eventually negative)
- Switch to CWT (Morlet wavelet) for better time-frequency resolution on transients
- Target industrial applications: bearing fault detection, drill noise separation in exploration
- Add objective metrics: PESQ, STOI (once training stabilizes)

## Current training status (as of March 2025)
- Running on SNR = +6 dB mixtures (speech audible but noise clearly present)
- U-Net with waveform loss activated
- First epochs in progress – stay tuned for denoised audio samples !

## Setup & Requirements

```bash
pip install torch torchaudio numpy matplotlib scipy pesq pystoi torchinfo
