#!/bin/bash
# update all jupyter notebooks

for notebook in *.ipynb
do
  echo updating $notebook
  jupyter nbconvert --to notebook --inplace --execute $notebook
done
