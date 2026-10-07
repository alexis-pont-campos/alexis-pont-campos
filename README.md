### Alexis Pont-Campos

*I'm still editing my profil, dont consider this as a final version*

I work on mathematical modelling, optimal control and scientific computing.

MSc in Mathematical Modelling and Applied Analysis (Université Savoie Mont-Blanc).
I spent a year in industry building a differentiable simulation and model predictive
control prototype in Python/JAX — deriving the model, implementing the solver, and
finding out where the two disagree.

Mostly interested in problems where a physical model, an optimisation problem and
working code have to be made to fit together.

---





<details>
<summary>Tools and topics</summary>

**Languages** — Python, C++, R, Matlab, SQL, LaTeX
**Scientific Python** — NumPy, SciPy, JAX, scikit-learn, OpenCV, Matplotlib
**Topics** — optimal control, model predictive control, automatic differentiation,
convex and non-convex optimisation, PDE modelling, image processing

</details>

---

Based in Rhône-Alpes, France. Open to engineering and research positions, and to PhD
projects (including CIFRE), in modelling, scientific computing and image processing.

📫 alexispontc@gmail.com







<p align="center">
  <img src="figures/banner.png" width="90%">
</p>

---

# Projects

This section shows a part of my topic of interest in term of coding, from pure optimisation problem to aestetic image prossesing or stuff that i simply find interresting.
Some of those sections are wildly inspired from the lectures and practical works i had in university, in particular **[Stéphane Breuils](https://github.com/sbreuils)** for image processing, **[Dorin Bucur](https://orcid.org/0000-0002-8331-8481)** and **[Laurent Vuillon](https://github.com/laurentvuillon)** for optimisation problems.

<!-- ▼▼ À REMPLIR : remplace ces quatre lignes par tes vrais sujets de TP.
     Garde le format : dossier / une phrase / les méthodes nommées. ▼▼ -->

| Study | What it does | Key methods |
|---|---|---|
| [01 — Image processing](01-filtering-and-denoising/) | Denoising, jpeg compression,  | Gaussian and median filters, bilateral filter, Fourier-domain filtering |
| [02 — Traveller problem](traveller/) | | |
| [03 — Other graphs optimisation problems](graphs/) | | |


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


## License

MIT — see [LICENSE](LICENSE).

<!-- ▼▼ Si certains TP contenaient du code de départ fourni par l'enseignant,
     ajoute une ligne ici, par exemple :
     "Starter code for study 02 was provided as part of a university course;
      all algorithm implementations are my own." ▼▼ -->
