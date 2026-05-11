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

## Current Status (v11a)

- **Architecture**: SimpleUNet with complex mask prediction (real + imaginary)
- **Key Innovation**: Frequency-dependent "Dog Bone" clamping inspired by the Speech Banana in audiology
- **Noise types**: Stationary and pseudo-stationary sounds from ESC-50 (rain, wind, engines, helicopter, etc.)
- **Training data**: 1 - 1.5s frames from LibriSpeech + ESC-50 mixtures at multiple SNR levels
- **Loss**: Hybrid (Complex Mask L1 + Waveform L1 + strong Phase Consistency Loss)

## Features

- Phase-aware complex masking to reduce musical noise and hoarseness
- Frequency-dependent clamping ("Dog Bone") for intelligent noise suppression
- Pre-computed STFTs for fast training
- Overlap-Add inference with optional mixture blending (`alph ≈ 0.05`)
- Clean, reproducible pipeline


## Results & Performance (v11aa - Banana Clamping)

### Objective Results (PESQ - Wideband)

#### Performance Summary - Female Voice + Helicopter (10 seconds)

<div align="center">

| SNR    | PESQ Noisy | PESQ Denoised<br>(Model Phase) | Improvement<br>(vs Noisy) | Notes |
|--------|------------|--------------------------------|---------------------------|-------|
| 0 dB   | 1.029      | **1.195**                      | +0.166                    | Very challenging |
| 6 dB   | 1.071      | **1.515**                      |  +0.445                    | Reasonable recovery |
| 12 dB  | 1.263      | **2.046**                      | +0.783             | Good |
| 15 dB  | 1.441      | **2.363**                      | +0.922                    | Good |

</div>

<br>

#### Performance Summary - Female Voice + Rain (10 seconds)

<div align="center">

| SNR     | PESQ Noisy | PESQ Denoised<br>(Model Phase) |  Improvement<br>(vs Noisy) | Notes |
|---------|------------|--------------------------------|--------------------------|-------|
| 0 dB    | 1.041      | **1.543**                      |  +0.503                    | Decent intelligibility |
| 6 dB    | 1.090      | **1.596**                      |+0.506                    | Reasonable recovery  |
| 12 dB   | 1.263      | **1.980**                      |  +0.718                    | Good |
| 15 dB   | 1.445      | **2.268**                      |  +0.823                    | Good |

</div>




### Visual Results & Audio Demos

All visual results (waveforms, spectrograms, and error maps) as well as audio examples have been saved in the `/results` folder.

**Included demonstrations:**
- **Female voice + Rain** at SNR = 0, 6, 12, and 15 dB
- **Female voice + Helicopter** at SNR = 0, 6, 12, and 15 dB

You can listen to the full 10-second clips and explore the corresponding STFT visualizations directly in the [results](results/) directory.

**Female voice + Rain - SNR 12 dB**


<div align="center">
  
| <p align="center"> <img src="results/Female_Rain_10s_SNR12dB_waveform.png" width="800" alt="Waveform"> </p> |
| --- |
| <p align="center"> <b><i> 10 seconds waveform: Female voice + Rain at SNR = 12 dB </i></b> </p> |

</div>

<br>

<div align="center">
  
| <p align="center"> <img src="results/Female_Rain_10s_SNR12dB_1s_STFT_analysis.png" width="800" alt="Waveform"> </p> |
| ---  | 
| <p align="center"> <b><i>  1 second frame: magnitude of STFT - female voice + Rain SNR= 12 dB </i></b> </p> |

</div>

<br>
<br>

**Female voice + Helicopter - SNR 12 dB**

<div align="center">
  
| <p align="center"> <img src="results/Female_Helico_10s_SNR12dB_waveform.png" width="800" alt="Waveform"> </p> |
| --- |
| <p align="center"> <b><i> 10 seconds waveform: Female voice + Helicopter at SNR = 12 dB </i></b> </p> |

</div>


<div align="center">
  
| <p align="center"> <img src="results/Female_Helico_10s_SNR12dB_1s_STFT_analysis.png" width="800" alt="Waveform"> </p> |
| ---  | 
| <p align="center"> <b><i>  1 second frame: magnitude of STFT - female voice + Helicopter SNR= 12 dB </i></b> </p> |

</div>





### Audio Examples (10 seconds)


**Female voice + Rain - SNR 12 dB**
- [▶️ Listen to Noisy](results/Female_Rain_10s_SNR12dB_noisy.wav)  
- [▶️ Listen to Denoised v11aa](results/Female_Rain_10s_SNR12dB_denoised.wav)  
- [▶️ Listen Clean](results/Female_Rain_10s_SNR12dB_clean.wav)


**Female voice + Helicopter - SNR 12 dB**
- [▶️ Listen to Noisy](results/Female_Helico_10s_SNR12dB_noisy.wav)
- [▶️ Listen to Denoised v11aa](results/Female_Helico_10s_SNR12dB_denoised.wav)
- [▶️ Listen to Clean](results/Female_Helico_10s_SNR12dB_clean.wav)
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
