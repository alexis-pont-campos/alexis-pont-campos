### Alexis Pont-Campos


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









---

# Projects

This section shows a part of my topic of interest in term of coding, from pure optimisation problem to aestetic image prossesing or stuff that i simply find interresting.
Some of those sections are wildly inspired from the lectures and practical works i had in university, in particular **[Stéphane Breuils](https://github.com/sbreuils)** for image processing, **[Dorin Bucur](https://orcid.org/0000-0002-8331-8481)** and **[Laurent Vuillon](https://github.com/laurentvuillon)** for optimisation problems.



| Project | What it does | Key methods |
|---|---|---|
| [01 — Image processing](01-filtering-and-denoising/) | Denoising, jpeg compression,  | Gaussian and median filters, bilateral filter, Fourier-domain filtering |
| [02 — Traveller problem](traveller/) | Optimal travel through random points | Elastic net and simulated annealing |
| [03 — Other graphs optimisation problems](graphs/) | Optimal path and cut algorithms | Floyd, Dijtra, Simplex method|
<<<<<<< HEAD
| [04 — Fluid simulation and shape optimisation (ongoing)](fluid_sim/) | Simulate a 2D fluid and find an optimal nose cone  | discretization of the Navier-Stokes equations and optim of the parameted shape|
=======
| [04 — Fluid simulation and shape optimisation (ongoing project)](fluid_sim/) | Simulate a 2D fluid and find an optimal nose cone  | discretization of the Navier-Stokes equations and optim of the parameted shape|
>>>>>>> 7d7720033e45345a6e5409d8d04f6876a58b3b73


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




## License

MIT [LICENSE](LICENSE).

