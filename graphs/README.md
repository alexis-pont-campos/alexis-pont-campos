# Operations Research

Classical operations research algorithms implemented from their mathematical
definitions in Python — no `networkx`, no `pulp`, no solver library doing the work
behind a one-line call.

**→ [`operations_research.ipynb`](operations_research.ipynb)** — the notebook renders
directly on GitHub, figures and outputs included. Nothing to install to read it.

---

## Contents

| Section | Problem | Methods |
|---|---|---|
| 1 | Storing a graph, and what each layout costs | Adjacency matrix, incidence matrix, CSR pointer/adjacency pair, degree identities |
| 2 | Who can reach whom, and can the graph be layered | Boolean matrix algebra, transitive closure, strongly connected components, rank function, BFS/DFS |
| 3 | Cheapest route between vertices | Dijkstra (single source), Floyd–Warshall (all pairs, negative weights) |
| 4 | Most throughput a network can carry | Ford–Fulkerson, residual networks, Edmonds–Karp, minimum cut |
| 5 | Optimising under linear constraints | Simplex on a tableau, standard form, slack variables, unboundedness |

Travelling salesman heuristics — greedy construction, simulated annealing, 2-opt —
are covered separately.

---

## Validation

Every result is asserted against a value known *before* the code existed, which makes
these genuine regression tests rather than a record of whatever the implementation
happened to produce.

| Section | Check |
|---|---|
| 1 | `Σ d⁺ = Σ d⁻ = m`; the compact form round-trips to the successor sets |
| 2 | Components `{1}, {2,3,4,5,6}, {7}`; ranks `0,1,1,2,3,2,4`; every arc increases the rank; circuits detected |
| 3 | Dijkstra labels `(0,7,2,5,10,14,16)`; the full 7×7 Floyd matrix including negative entries; all 40 reconstructed paths real and correctly valued; the two algorithms agree where both apply |
| 4 | Max flow 12, with capacity bounds and conservation checked at every arc and vertex; minimum cut `{A,C}` of capacity 12 |
| 5 | Optimum 36000 at `(200,600)`; a second instance cross-checked against `scipy.optimize.linprog` |

---

## A few implementation choices

**Boolean algebra instead of counting.** Replacing `(+, ×)` by `(or, and)` in the
matrix product turns `A^k` from a count of walks into a reachability test, so the
transitive closure is `(I + A)^(n−1)` in a single expression and the strongly connected
components are the agreement between `R` and its transpose.

**Next-hop instead of pivot vertices.** Floyd–Warshall is usually presented with a
matrix of pivot vertices and a recursive path reconstruction. Storing the next hop
carries the same information and makes reconstruction a plain loop.

**Certificates, not just values.** Ford–Fulkerson returns the minimum cut alongside the
flow. Because the cut is read off the residual network at optimality, its capacity
necessarily equals the flow — the max-flow min-cut theorem falls out of the
construction instead of being proved separately.

**Refusal over silent error.** Dijkstra raises on negative weights rather than
returning a plausible wrong answer; Floyd raises on a negative circuit; the simplex
raises on an inadmissible slack basis instead of pretending to have a phase-I.

---

## Running it

```bash
pip install numpy matplotlib scipy
jupyter notebook operations_research.ipynb
```

`scipy` is only used to cross-check the simplex; the notebook runs without it.

---

## Scope

Strongly connected components go through the reachability matrix in `O(n³)` rather than
Tarjan's linear-time algorithm — the matrix formulation is the one being illustrated,
and at this scale the difference is invisible. Bellman–Ford, the missing middle between
Dijkstra and Floyd, is not implemented. The simplex has no phase-I and no anti-cycling
rule, so degenerate programs are capped rather than guaranteed to terminate.

The thread worth pulling: maximum flow *is* a linear program, and its dual *is* the
minimum cut. Sections 4 and 5 are the same theorem seen from two sides.

---

The algorithms are classical and the worked instances come from a university operations
research course; the implementations, figures and validation are my own.
