# Transport Topology for Fixed-Support Sinkhorn Layers

Fixed-support transport operators, support constructors, and certificate validators.

## Repository layout

- `src/`: reference implementation.
- `docs/results/`: retained experimental evidence.
- `paper/`: corrected PDF, LaTeX sources, and arXiv source ZIP.
- `docs/VALIDATION.md`: verification results and outstanding discrepancies.

## Environment

Use the inherited pinned environment on NixOS:

```sh
nix-shell
```

In a compatible Python environment, install with `uv sync`.
The pinned Nix environment defines historical numerical replay; newly resolved
package versions may produce different floating-point results.

## Manuscript

```sh
sh paper/build.sh
```

See `paper/README.md` for the correction notice. The supplied PDF is preserved;
the source ZIP is synchronized with the cleaned LaTeX sources.

## License

MIT; see [LICENSE](LICENSE).
