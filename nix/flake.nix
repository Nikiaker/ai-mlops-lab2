{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: 
  let 
    system = "x86_64-linux";
    pkgs = import nixpkgs { system = "x86_64-linux"; config.allowBroken = true; };
    fetchurl = pkgs.fetchurl;
    packageOverrides = pkgs.callPackage ./python-packages.nix { inherit pkgs fetchurl; };
    pythonCustom = pkgs.python3.override { inherit packageOverrides; };
  in 
  {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        python3
        python3Packages.pip

        python3Packages.torch
        python3Packages.torchvision
        python3Packages.pytorch-lightning
        python3Packages.wandb
        python3Packages.optuna
        #python3Packages.bentoml
        ];

        shellHook = ''
          SOURCE_DATE_EPOCH=$(date +%s)
          VENV=.venv

          if test ! -d $VENV; then
            python3.12 -m venv $VENV
          fi
          source ./$VENV/bin/activate
          export PYTHONPATH=`pwd`/$VENV/${pkgs.python3.sitePackages}/:$PYTHONPATH
          pip install -r requirements.txt
        '';

        postShellHook = ''
          ln -sf ${pkgs.python3.sitePackages}/* ./.venv/lib/python3.12/site-packages
        '';
    };
  };
}
