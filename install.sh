# set up environment
conda create -y -n leichte_sprache python=3.11
conda activate leichte_sprache
pip install -e .

# install pre-commit hook
pre-commit install

# install German lcoale (needed for crawling)
sudo locale-gen de_DE.utf8

# set up data dir
mkdir -p data