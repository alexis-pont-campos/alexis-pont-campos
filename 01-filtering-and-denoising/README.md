# Filtering and Denoising

An introduction to deep learning through image processing. The classical operators —
convolution, morphology, Sobel, Hough, Tikhonov and Wiener, the JPEG transform — are
implemented from their mathematical definitions in NumPy, and each one is paired with the
deep-learning layer or idea it became. A single algorithm is imported rather than
rewritten: a pretrained depth network, the starting point of the mini-project.

**→ [`filtering_and_denoising.ipynb`](filtering_and_denoising.ipynb)** — the notebook
renders directly on GitHub, figures and outputs included. Nothing to install to read it.

---

## Contents

| Section | Problem | Methods |
|---|---|---|
| 1 | Smoothing and sharpening, and what a kernel does to each frequency | Convolution as a sliding dot product (im2col), convolution theorem, box vs Gaussian response, separability, unsharp masking |
| 2 | Cleaning up a binary shape | Erosion, dilation, opening, closing, morphological gradient; max-pooling as a dilation with stride |
| 3 | Finding edges — and learning the filter that finds them | Sobel gradient, magnitude and orientation; a 3×3 kernel fitted by gradient descent |
| 4 | Finding straight lines | Hough voting in (ρ, θ), peaks by non-maximum suppression |
| 5 | Undoing a blur without amplifying the noise | Tikhonov in Fourier, Wiener as the minimum mean-square-error filter, Wiener derived as Tikhonov with the noise-to-signal ratio as penalty, power-law image prior |
| 6 | Mini-project: moving the focus of a photo after it was taken | Monocular depth (MiDaS) → 3-D point cloud → novel views by z-buffer splatting → thin-lens circle of confusion, layered bokeh |
| 7 | Compressing an image | 8×8 DCT-II, IJG quantisation tables, rate vs distortion |
| 8 | Bonus: colour and the "HDR" look | sRGB and linear light, luminance, histogram equalisation per channel, on luminance, and local (CLAHE) |

Each section names its deep-learning counterpart: convolution layers, max-pooling,
learned edge filters, the Deep Hough Transform, weight decay as a prior, learned depth,
learned codecs, normalisation layers.

---

## Validation

Every check is an `assert` in the notebook, made against something known independently
of the code under test: a closed form, a reference implementation, or measured ground
truth.

| Section | Check |
|---|---|
| 1 | Direct convolution = FFT convolution = `scipy.ndimage.convolve`; the 2-D Gaussian blur = two 1-D passes |
| 2 | A 20×20 square erodes to 18² and dilates to 22² pixels; erosion/dilation duality; opening idempotent, opening ⊆ x ⊆ closing; grey erosion = `scipy.ndimage.grey_erosion`; dilation with stride 2 = 2×2 max-pooling, exactly |
| 3 | Sobel returns the exact slopes of a linear ramp; gradient descent on (noise, Sobel(noise)) pairs recovers the kernel to 10⁻⁶ |
| 4 | Four synthetic lines found at their exact `(ρ, θ)`, each peak equal to the line's pixel count; accumulator bit-identical to `skimage.transform.hough_line` |
| 5 | Fourier Tikhonov = dense solve of the normal equations; Wiener with a prior fitted on the degraded image alone reaches 26.64 dB vs 26.62 dB for the best hand-tuned λ (degraded 23.49 dB, oracle ceiling 27.13 dB) |
| 6 | Predicted inverse depth vs stereo ground truth: Pearson r = 0.825; projection ∘ back-projection = identity; zero aperture returns the photo; the patch at the focus point comes back unchanged while the other keeps 16–38 % of its sharpness |
| 7 | DCT matrix orthonormal and equal to `scipy.fft.dctn`; PSNR within 0.011 dB of Pillow/libjpeg from q = 5 to 95 |
| 8 | Equalised luminance uniform (Kolmogorov distance 0.003); matches `skimage.exposure.equalize_hist` to 0.01 |

---

## A few implementation choices

**One sliding window for everything.** `sliding_window_view` exposes every patch of the
image as a view, without copying. Contracted with a kernel it is a convolution (§1);
reduced by `min` or `max` over a boolean mask it is morphology (§2); read as a design
matrix it is what gradient descent needs to learn the kernel back (§3). This is im2col,
the way convolution layers are actually computed.

**Wiener is literally a call to Tikhonov.** `wiener(y, H, Sx, Sn)` returns
`tikhonov(y, H, 1, Sn / Sx)` — the derivation of §5 made executable. The image prior is
fitted on the degraded image through `S_y = |H|² S_x + σ²`, so the restoration uses
neither the clean image nor a tuned λ, and still matches the best λ of a sweep that
uses both.

**Blur lives in inverse depth and in linear light.** The circle of confusion of a thin
lens is linear in `1/Z`, the quantity the depth network predicts, so the scene is sliced
in equal steps of `1/Z`. Each layer is blurred in gamma-decoded values with premultiplied
alpha and composited back to front: highlights bloom into discs, and a blurred
foreground spills over the background, never the reverse.

**Matching the reference, not approximating it.** The Hough accumulator reproduces
scikit-image's angle grid, offset and rounding, so the two agree bit for bit. The JPEG
codec uses the same IJG quality scaling as libjpeg, so the PSNR matches Pillow at every
quality; the remaining 0.01 dB is libjpeg's integer DCT.

---

## Running it

```bash
pip install numpy scipy matplotlib scikit-image pillow pooch onnxruntime
jupyter notebook filtering_and_denoising.ipynb
```

On first run, §6 downloads the MiDaS v2.1 small weights (67 MB in ONNX format, cached by
`pooch`) and the Middlebury motorcycle pair (cached by scikit-image). `onnxruntime` runs
the network on CPU — no PyTorch, no GPU — and is only needed for §6. `scipy` and Pillow
are only used as references; scikit-image supplies the test images, the resize around
the network, and CLAHE. The whole notebook runs in under a minute on a laptop.

---

## Scope

Filters run as sliding windows, `O(k²)` per pixel; for large kernels the FFT path of §1
wins, and the two agree. The Hough transform finds infinite lines only — no segments, no
probabilistic variant. Section 5 assumes periodic boundaries and a known blur and noise
level; the oracle Wiener uses the clean image's spectrum and is only there as a ceiling.
MiDaS predicts inverse depth up to scale and shift, so the 1.5–4.5 m range is chosen,
not measured; the renderer does not inpaint what the foreground hides, which is where
the novel views leave holes. The JPEG codec is grayscale, without chroma subsampling or
entropy coding — file sizes come from Pillow. CLAHE is imported from scikit-image.

The thread worth pulling: every restoration in §5 is one fixed linear layer whose
weights are set by a prior, and §3 shows the same layer learned from examples instead.
Stack such layers with non-linearities — the max of §2 is one — and train them on
(noisy, clean) pairs: that is a CNN denoiser, DnCNN (Zhang et al., 2017).

---

The methods are classical. The test images come from scikit-image's `data` module — the
motorcycle and its ground-truth disparity from the Middlebury 2014 stereo dataset — and
the depth network is MiDaS v2.1 small (Ranftl et al., 2020).
