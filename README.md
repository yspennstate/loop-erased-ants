# Loop-erased ants need not find shortest paths

Yitzchak Shmalo

In the ant process an ant walks from a nest to a food source on a weighted graph, and the
edges of the loop erasure of its walk gain weight one. Kious, Mailler and Schapira conjectured
that the ants find the shortest paths when the walk is erased backward; Armfield and Mailler
conjectured the same for the forward erasure. The two rules give the reinforced path the same
law, and on a graph with six vertices an edge on no shortest path keeps a positive limiting
weight with positive probability. Adding a path at the nest gives counterexamples at every
distance at least three.

- `main.tex`, `main.pdf`: the paper.
- `check.py`: exact rational checks of the computations in the paper (the sum of the
  path numerators, the drift, the polynomial bound used in the trapping argument) and a comparison of the law of
  the reinforced path computed from spanning trees, from the forward erasure and from the
  backward erasure. Requires Python 3 with `sympy`; run `python check.py`.
