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


## Results & Performance (v11aa - Banana Clamping)

### Objective Results (PESQ - Wideband)

| Condition                | SNR    | PESQ Noisy | PESQ Denoised | Improvement | Listening Quality |
|--------------------------|--------|------------|---------------|-------------|-------------------|
| Female + Rain            | 6 dB   | 1.090      | **1.596**     | +0.506      | Good              |
| Female + Rain            | 12 dB  | 1.263      | **1.980**     | +0.718      | Very Good         |
| Female + Helicopter      | 12 dB  | 1.263      | **2.046**     | +0.783      | Very Good         |
| Female + Helicopter      | 15 dB  | 1.445      | **2.268**     | +0.823      | Excellent         |

### Visual Results

**Female + Rain - SNR 12 dB**

| ![Female + Rain - Waveform](results/Female_Rain_10s_SNR12dB_waveform.png)  |
| --- |
| ![Female + Rain - Spectrogram](results/Female_Rain_10s_SNR12dB_1s_STFT_analysis.png) |


**Female + Helicopter - SNR 12 dB**

![Female + Helicopter - Spectrogram](results/Female_Helico_10s_SNR12dB_STFT.png)  
![Female + Helicopter - Waveform](results/Female_Helico_10s_SNR12dB_waveform.png)

### Audio Examples (10 seconds)

- **[Female + Rain - SNR 12 dB - Noisy](results/Female_Rain_10s_SNR12dB_noisy.wav)**  
- **[Female + Rain - SNR 12 dB - Denoised](results/Female_Rain_10s_SNR12dB_denoised.wav)**

- **[Female + Helicopter - SNR 12 dB - Noisy](results/Female_Helico_10s_SNR12dB_noisy.wav)**  
- **[Female + Helicopter - SNR 12 dB - Denoised](results/Female_Helico_10s_SNR12dB_denoised.wav)**

---

**Key Takeaway**:  
The **Banana Clamping** ("Os de Chien") strategy delivers strong and consistent performance across different noise types and SNR levels, with particularly good voice clarity and harmonic preservation.







## ---------------- under construction
    
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
