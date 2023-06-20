# Flowformer

Automatic detection of blast cells in ALL data using transformers. 

Official implementation of our work: *"Automated Identification of Cell Populations in Flow Cytometry Data with Transformers"*
by Matthias Wödlinger, Michael Reiter, Lisa Weijler, Margarita Maurer-Granofszky, Angela Schumich, Elisa O Sajaroff, Stefanie Groeneveld-Krentz, Jorge G Rossi, Leonid Karawajew, Richard Ratei and Michael Dworzak

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

(If you run into issues with newer versions of dependencies check the `requirements.txt` file. It contains the environment package dependencies at the time of testing)

IMPORTANT: As of now the flowmepy package is only supported on windows. If you are running a unix based system and want to try out our method you will need to preload the data (for example to a pandas dataframe) on a windows machine and then adapt the lines in the code where the flowme python package is called. Simply load your preloaded event matrices (dataframes or csv) instead of the `events = sample.events()` lines and load your gate label matrices (dataframes or csv) instead of the lines where `labels = sample.gate_labels()`. Sorry for the inconvenience, we are working on a solution.

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

If selected, the fcs data is preloaded in the tmp folder as torch tensors. This speeds up the loading process.

For example, to test the pretrained model in `saved/models/vie14_bue/0720_153309` the command is 
```
python test.py -c saved/models/vie14_bue/0720_153309/config.json -r saved/models/vie14_bue/0720_153309/model_best.pth -d GPU_IDX
```

### Predict
To apply a trained model to a single sample use the `predict.py` script:
```
python predict -c saved/models/EXP_NAME/TIMESTAMP/config.json -r saved/models/EXP_NAME/TIMESTAMP/model_best.pth -d GPU_IDX -f PATH/TO/XML/OR/ANALYSIS/FILE
```

The script computes the metric scores for the given file and plots a standard panel of marker combinations. The panel shows blasts in red an other cells in blue. The top row shows the ground truth information and the bottom row the predicted information. If the script is run with the optional option `-p` every possible marker combination is plotted as well (this might take around 30s, depending on the number of markers in the config file).

For example to predict the file `Bue072d15` from the bue dataset you need to run:
```
python predict.py --config config/vie14_bue.json --resume saved/models/vie14_bue/0720_153309/model_best.pth -d 3 --file /PATH/TO/Bue072d15.xml -p
```

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

## Erratum

The default config uses 3 hidden layers which results in 4 ISAB layers in total contrary to 3 as stated in the paper. There is a typo in Table 2 of the paper: The median f1 score of our method when trained on vie20 and tested on vie14 is actually 0.95 (and the mean f1 score is 0.84), making our method better than the GMM based method [14] in every benchmark.

## Citation

If you use this project please consider citing our work

```
@article{wodlinger2022automated,
  title={Automated identification of cell populations in flow cytometry data with transformers},
  author={W{\"o}dlinger, Matthias and Reiter, Michael and Weijler, Lisa and Maurer-Granofszky, Margarita and Schumich, Angela and Sajaroff, Elisa O and Groeneveld-Krentz, Stefanie and Rossi, Jorge G and Karawajew, Leonid and Ratei, Richard and others},
  journal={Computers in Biology and Medicine},
  volume={144},
  pages={105314},
  year={2022},
  publisher={Elsevier}
}
```
