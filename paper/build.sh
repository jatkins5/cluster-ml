#!/bin/bash
#SBATCH -p batch -N 1 -n 1 -c 1 --mem=4G -t 00:15:00
#SBATCH -J paper
#SBATCH -o build_%j.log
# Build the draft. MNRAS's class ships with TeX Live; if this installation
# lacks it, fetch mnras.cls / mnras.bst from CTAN into this folder.
cd /oscar/data/idellant/cluster-ml/paper
module load texlive/20240312-xnyd
if ! kpsewhich mnras.cls >/dev/null; then
  echo "mnras.cls not in TeX Live here; fetching from CTAN"
  curl -sSL -o mnras.zip https://mirrors.ctan.org/macros/latex/contrib/mnras.zip
  unzip -jo mnras.zip 'mnras/mnras.cls' 'mnras/mnras.bst' && rm -f mnras.zip
fi
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > latexmk.out 2>&1
echo "exit $?"
grep -E "^! |Warning: Citation|Warning: Reference|Output written" main.log | sort | uniq -c | head -20
