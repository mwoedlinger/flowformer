# Flowformer

Automatic detection of blast cell using transformer based neural networks.

## Installation

The package needs flowme and torch to be installed:
- Flowme: https://smithers.cvl.tuwien.ac.at/kwc/dev/flowme-python-api
- Torch: https://pytorch.org/get-started/locally/

Installation with setup.py:

```
python -m venv env
source env/bin/activate
python setup.py install
```

## Usage

The config is set in a config file (check out the config folder for some examples. config_setT.json should be the go-to template file).
The script should be self explanatory: `train.py` is for training, `test.py` for testing (also created an mrd plot) and `predict.py` for applying the model to a specific file (can also be used to generate the typical marker projectio plots).

## Project layout

The project is based on [this](https://github.com/victoresque/pytorch-template) pytorch template. For further questions mail mwoedlinger@cvl.tuwien.ac.at.