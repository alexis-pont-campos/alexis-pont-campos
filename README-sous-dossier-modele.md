# 01 — Filtering & Denoising

<!-- Ce fichier est un GABARIT. Copie-le dans chaque sous-dossier,
     renomme-le README.md, et adapte les 5 sections.
     Vise 10 lignes de texte maximum + 1 ou 2 figures. Pas plus. -->

Restoring a noisy image while preserving the structures that matter.

## Problem

<!-- 2-3 phrases. Quel est le problème, et pourquoi il n'est pas trivial. -->

Additive Gaussian noise degrades an image uniformly, but naive smoothing degrades edges just as
uniformly. The difficulty is separating high-frequency content that is noise from high-frequency
content that carries information.

## Method

<!-- 3-5 phrases. Nomme les méthodes. C'est la section que lit un ingénieur. -->

Three approaches are compared. A Gaussian filter applies isotropic smoothing through convolution
with a separable kernel. A median filter replaces each pixel by the median of its neighbourhood,
which suppresses impulse noise without the blurring introduced by linear averaging. A bilateral
filter weights neighbours by both spatial and intensity distance, which preserves edges at the
cost of a non-separable, more expensive computation.

## Results

<p align="center">
  <img src="figures/comparison.png" width="90%">
</p>

<!-- Une phrase de commentaire sur ce qu'on voit. Un chiffre si tu en as un : PSNR, SSIM, temps. -->

The bilateral filter gives the best edge preservation (PSNR 31.2 dB vs. 28.4 dB for the Gaussian
filter at equivalent noise reduction), at roughly 8× the runtime.

## Running

```bash
python main.py
```

## Notes

<!-- Optionnel, mais ça inspire confiance : ce qui est limité, ce que tu ferais ensuite. -->

The bilateral filter implementation is a direct one, without the histogram-based acceleration
used in production libraries; it is intended to be readable rather than fast.
