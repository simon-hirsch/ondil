{
  description = "Ondil Flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/a1bab9e494f5f4939442a57a58d0449a109593fe";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
      };
      pyproject = fromTOML (builtins.readFile ./pyproject.toml);
    in
    {
      packages.${system} = rec {
        default = pkgs.python3.pkgs.callPackage (./package.nix) {
          self = self;
          lib = pkgs.lib;
          pyproject = pyproject;
        };
        ondil = default;
      };

      devShells.${system}.ondil = pkgs.mkShell {
        name = "Ondil-dev-shell";
        venvDir = "./.venv";
        buildInputs = with pkgs.python3.pkgs; [
          self.packages.${system}.default
        ];
      };
    };
}
