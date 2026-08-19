# Image Processing

Classical image processing algorithms implemented from scratch in Python — filtering and denoising, segmentation, mathematical morphology, and edge detection.

Each folder is a self-contained study: the problem, the method, the implementation, and the results.

<!-- ▼▼ À REMPLIR : mets ici une image large qui montre 3-4 de tes meilleurs résultats côte à côte.
     C'est la première chose qu'on voit. Fais-la sous matplotlib (subplots) et exporte en PNG. ▼▼ -->

<p align="center">
  <img src="figures/banner.png" width="90%">
</p>

---

## Contents

<!-- ▼▼ À REMPLIR : remplace ces quatre lignes par tes vrais sujets de TP.
     Garde le format : dossier / une phrase / les méthodes nommées. ▼▼ -->

| Study | What it does | Key methods |
|---|---|---|
| [01 — Filtering & denoising](01-filtering-and-denoising.md/) | Removing noise while preserving edges | Gaussian and median filters, bilateral filter, Fourier-domain filtering |
| [02 — Segmentation](02-segmentation/) | Partitioning an image into meaningful regions | Otsu thresholding, region growing, k-means, watershed |
| [03 — Mathematical morphology](03-mathematical-morphology/) | Shape-based analysis of binary and grayscale images | Erosion, dilation, opening/closing, skeletonization, granulometry |
| [04 — Edge detection](04-edge-detection/) | Locating and characterizing intensity discontinuities | Sobel and Prewitt operators, Laplacian of Gaussian, Canny |

---

## Results

<!-- ▼▼ À REMPLIR : 2 ou 3 comparaisons avant/après. Une image vaut tout le texte ci-dessus.
     Si tu n'as qu'une figure, mets-en une seule — mieux vaut une bonne qu'un remplissage. ▼▼ -->

**Denoising — Gaussian vs. bilateral filter**

<p align="center">
  <img src="figures/denoising-comparison.png" width="90%">
</p>

**Segmentation — Otsu vs. watershed**

<p align="center">
  <img src="figures/segmentation-comparison.png" width="90%">
</p>

---

## Approach

The algorithms here are implemented directly from their mathematical definitions using NumPy,
rather than called from a library. Library implementations (OpenCV, scikit-image) are used only
as a reference to validate the output.

The point is to make the underlying operations explicit: convolution kernels are built by hand,
morphological operators are written in terms of structuring elements, and the frequency-domain
methods are derived from the discrete Fourier transform rather than treated as black boxes.

---

## Stack

Python 3.11 · NumPy · SciPy · Matplotlib · OpenCV (reference only)

---

## Running the code

```bash
git clone https://github.com/YOUR-USERNAME/image-processing.git
cd image-processing
pip install -r requirements.txt
```

Each study runs independently:

```bash
cd 01-filtering-and-denoising
python main.py
```

Figures are written to the `figures/` folder of the corresponding study.

<!-- ▼▼ À REMPLIR : adapte les noms de fichiers ci-dessus à ta vraie arborescence.
     Si tes TP sont des notebooks, remplace par : jupyter notebook 01-filtering-and-denoising/ ▼▼ -->

---

## About

MSc in Mathematical Modelling and Applied Analysis, Université Savoie Mont-Blanc.

Background in optimal control, optimization and scientific computing, with a year in industry
working on differentiable simulation and model predictive control in Python/JAX.

Currently looking for engineering or research positions in modelling, scientific computing
and image processing in the Rhône-Alpes region.

<!-- ▼▼ À REMPLIR : ton email, et ton LinkedIn si tu en as un. ▼▼ -->

📫 your.email@example.com

---

## License

MIT — see [LICENSE](LICENSE).

<!-- ▼▼ Si certains TP contenaient du code de départ fourni par l'enseignant,
     ajoute une ligne ici, par exemple :
     "Starter code for study 02 was provided as part of a university course;
      all algorithm implementations are my own." ▼▼ -->
