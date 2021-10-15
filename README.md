# Flowformer

Automatic detection of blast cells in ALL data using transformers.

## Installation

Install the flowformer package and all required packages with the setup.py file:
```
python -m pip install .
```

The package needs flowmepy to be installed. flowmepy is a python package for fcm data loading. For more information see: https://pypi.org/project/flowmepy/
Install with:
```
pip install flowmepy
```

## Usage

The training configurations is defined in a config json file (check out the config folder for some the configs used for the experiments in our paper).
The scripts should be self explanatory: `train.py` is for training and `test.py` for testing (also creates an mrd plot).

### Training
Example for training of experiment specified in config/vie.json on gpu with index 0:
```
python train.py -c config/vie.json -d 0
```
The trained models is then saved under `saved/models/vie/TIMESTAMPT/model_best.pth`.

### Testing
Example for testing the model the model trained in the example above:
```
python test.py -c saved/models/vie/TIMESTAMPT/config.json -r saved/models/vie/TIMESTAMPT/model_best.pth -d 0
```
The test output is printed to the console and additional information is saved under `saved/models/vie/TIMESTAMPT/model_best.pth`.

If selected, the fcs data is preloaded in the tmp folder as torch tensors. This speeds up the loading process

### Data
the config file expects a `data_dir` in the dataloader args. The specified data_dir is supposed to contain three text files (train.txt and eval.txt are needed during training and test.txt is needed during testing), where every line contains the path to a fcm file (`.xml` or `.analysis`). The vie14, bln and bue data from our work can be downloaded from here: https://flowrepository.org/id/FR-FCM-ZYVT
If you want to reproduce the experiments simply changes the path in the corresponding data txt files and run the experiments from the config folder.

### Run your own experiments
Here is what needs to be done (I recommend starting with one of the existing config files and then modifying the sections specified below):
- change "name" to an experiment name of your choice
- specifiy the "data_dir" to a directory containing train.txt, eval.txt and test.txt files with paths to your data
- modify the marker string to fit your data. The string is a sequence of markers seperated by a single comma and space.

## Project layout

The project is based on [this](https://github.com/victoresque/pytorch-template) pytorch template.