# set up environment
conda create -n leichte_sprache python=3.11
conda activate leichte_sprache
pip install -r requirements.txt

# install pre-commit hook
pre-commit install