
# Quick start (Windows PowerShell):

## conda
conda --version
conda info --envs
conda activate base
python --version

## Conda setup
### install conda and check version with 
conda --version

### create an environment file (environment.yml)
conda env create -f environment.yml  

### to update conda environment
conda env update -f environment.yml --prune  

### To activate this environment  ( if project is existing)
conda activate poisson-regression    

### run code
python poisson-regression.py  

### To install additional libs
conda install tqdm -n poisson-regression  

 