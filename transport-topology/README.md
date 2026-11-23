# Support Topology and Gradient Mixing in Sinkhorn Layers

Fixed-support Sinkhorn differential operators, sparse support generators, contraction bounds, and numerical verification checks.

## Repository layout

- `src/`: reference operators (`transport_topology.py`) and sparse support constructors (`topology_masks.py`).
- `scripts/verify_paper_examples.py`: deterministic numerical checks reproducing the finite-difference VJP, canonical support comparison, face-lattice stress test, and closed-form formula verifications in the paper.
- `paper/`: manuscript sources and compiled PDF (`main.tex`, `main.pdf`, `references.bib`, `research_preprint.sty`, `build.sh`).

## Environment

Use the pinned Nix environment:

```sh
nix-shell
```

Or install Python dependencies with `uv sync`.

## Verification and manuscript build

```sh
PYTHONPATH=. python3 scripts/verify_paper_examples.py
sh paper/build.sh
```

## License

MIT; see [LICENSE](LICENSE).
