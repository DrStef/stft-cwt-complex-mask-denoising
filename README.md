# Advanced Speech Denoising with Complex Masks STFT, CWT

<br> 

**Dr. Stéphane Dedieu** 
<br>Applied Mathematics | Digital Signal Processing | ML  <br>
January - March 2026  <br>
<a href="https://www.linkedin.com/in/sdedieu/">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" width="30" height="30">
</a>

<br>


## Project Overview

This repository explores **high-fidelity speech denoising** using **complex ratio masking** (cIRM) in the time-frequency domain.  

The current focus is on **STFT-based processing** with a lightweight **SimpleUNet** and a custom **frequency-dependent clamping strategy** ("Dog Bone" / Banana profile). This approach aims to deliver strong noise reduction while preserving natural speech quality, particularly on female voices.

Future work will extend the pipeline to **Continuous Wavelet Transform (CWT)** for better handling of non-stationary and impulsive noises.

## Current Status (v07r)

- **Architecture**: SimpleUNet with complex mask prediction (real + imaginary)
- **Key Innovation**: Frequency-dependent "Dog Bone" clamping inspired by the Speech Banana in audiology
- **Noise types**: Stationary and pseudo-stationary sounds from ESC-50 (rain, wind, engines, helicopter, etc.)
- **Training data**: 1 - 1.5s frames from LibriSpeech + ESC-50 mixtures at multiple SNR levels
- **Loss**: Hybrid (Complex Mask L1 + Waveform L1 + strong Phase Consistency Loss)
- **Best validation loss**: ~0.387 (as of May 5, 2026)

## Features

- Phase-aware complex masking to reduce musical noise and hoarseness
- Frequency-dependent clamping ("Dog Bone") for intelligent noise suppression
- Pre-computed STFTs for fast training
- Overlap-Add inference with optional mixture blending (`alph ≈ 0.05`)
- Clean, reproducible pipeline

## Repository Structure (to be cleaned)

- `notebooks/` → Main experiments (v07r currently active)
- `precomputed_stft_v07r.pt` → Pre-computed STFTs for fast training
- `best_unet_denoiser_v07r.pth` → Best model checkpoint

## Next Steps

- Finalize post-processing and demo examples (10s clips)
- Create clean comparison tables and audio samples
- Prepare professional GitHub presentation
- Extend to CWT for non-stationary noise
- Industrial applications (EERS, bearing noise, MRI scanner, etc.)

---

**License**: MIT  
**Status**: Active research & development




## Setup & Requirements

```bash
pip install torch torchaudio numpy matplotlib scipy pesq pystoi torchinfo
