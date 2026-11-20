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
      python-pkgs.numpy
      python-pkgs.jax
      python-pkgs.jaxlib
      python-pkgs.pytest
      python-pkgs.ruff
    ]))
    pkgs.tectonic
    pkgs.poppler-utils
    pkgs.git
  ];
}

