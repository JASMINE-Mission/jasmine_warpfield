#!/bin/bash
# update all jupyter notebooks

# exit on error
set -e

for notebook in *.ipynb
do
  echo updating $notebook
  jupyter nbconvert --to notebook --inplace --execute $notebook
done
