{ pkgs ? import <nixpkgs> {} }:
let
  fetchurl = pkgs.fetchurl;
  packageOverrides = pkgs.callPackage ./python-packages.nix { inherit pkgs fetchurl; };
  pythonCustom = pkgs.python3.override { inherit packageOverrides; };
in
  pkgs.mkShell {
    packages = with pkgs; [
        python3
        python3Packages.pip

        python3Packages.torch-bin
        python3Packages.torchvision-bin
        python3Packages.pytorch-lightning
        python3Packages.wandb
        python3Packages.optuna
        python3Packages.boto3
        awscli        
        #(pythonCustom.withPackages(p: [ p.bentoml ]))
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
}