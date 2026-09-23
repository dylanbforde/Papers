let
  pinnedNixpkgs = builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/b6018f87da91d19d0ab4cf979885689b469cdd41.tar.gz";
    sha256 = "sha256-twXPFqFsrrY5r28Zh7Homgcp2gUMBgQ6WDS98Q/3xFI=";
  };
in
{ pkgs ? import pinnedNixpkgs {} }:

pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.ipykernel
      python-pkgs.matplotlib
      python-pkgs.biopython
      python-pkgs.requests
      python-pkgs.numpy
      python-pkgs.scipy
      python-pkgs.jax
      python-pkgs.jaxlib
      python-pkgs.transformers
      python-pkgs.torch
      python-pkgs.optax
      python-pkgs.pytest
      python-pkgs.ruff
    ]))
    pkgs.ripgrep
    pkgs.git
    pkgs.poppler-utils
    (pkgs.texlive.withPackages (tex-pkgs: with tex-pkgs; [
      scheme-small
      enumitem
    ]))
  ];
}
