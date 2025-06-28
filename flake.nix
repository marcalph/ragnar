{
  description = "C++ Dev Shell for macOS (Apple Silicon)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";

  outputs = { self, nixpkgs }:
    let
      system = "aarch64-darwin";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          pkgs.clang
          pkgs.libcxx
          pkgs.xz  # if needed for compression or build deps
        ];

        shellHook = ''
          export CC=clang
          export CXX=clang++
          # [[ -n $ZSH_VERSION ]] && source ~/.zshrc
        '';
      };
    };
}
