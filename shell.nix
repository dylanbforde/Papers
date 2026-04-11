{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.ipykernel
      python-pkgs.matplotlib
      python-pkgs.biopython
      python-pkgs.requests
      python-pkgs.numpy
      python-pkgs.jax
      python-pkgs.jaxlib
      python-pkgs.transformers
      python-pkgs.torch
      python-pkgs.optax
    ]))
  ];
}
