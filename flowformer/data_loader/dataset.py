import torch
import pandas as pd
from random import randint
from pathlib import Path
from shutil import rmtree
from tqdm import tqdm
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from ..utils import get_project_root, suppress_stdout_stderr, read_stats

with suppress_stdout_stderr():
    import flowme



class FlowData(torch.utils.data.Dataset):
    """
    Flow cytometry pytorch dataset. The experiment data is stored in xml and fcs files. The fcs files contain the cell data while the xml
    files contain informations regarding the specific experiment. if fast_preload is true during initialization of the object the fcs data
    is preloaded and saved as pickled torch tensors in the folder determined by fast_preload_dir. The folder is deleted when __del__ ist called.
    """

    def __init__(self, **kwargs):
        assert kwargs['data_type'] in ['train', 'eval', 'test'], \
            'data_type must be either "train", "eval" or "test" but got {}'.format(
                kwargs['data_type'])

        self.data_type = kwargs['data_type']
        self.marker_list = kwargs['markers'].replace(' ', '').split(',')

        # If data dir is specified it will look for train.txt, eval.txt and test.txt, otherwise use it is expected that the
        # the text files are given directly with kwargs['train], kwargs['eval'] and kwargs['test'].
        if 'data_dir' in kwargs:
            self.data_root = get_project_root() / Path(kwargs['data_dir'])
            self.data_list = self.data_root / Path(kwargs['data_type'] + '.txt')
        else:
            self.data_list = get_project_root() / Path(kwargs[self.data_type])


        # Load files
        self.fast_preload = kwargs['fast_preload'] and (
            self.data_type != 'test')
        if self.fast_preload:
            Path(kwargs['fast_preload_dir']).mkdir(
                parents=True, exist_ok=True)
            self.tmp_path = Path(
                kwargs['fast_preload_dir']) / Path(self.data_type)
            self.files = self._preload()
        else:
            self.files = self._get_list()

        print('Created {} dataset with {} files from {}.'.format(
            kwargs['data_type'], len(self.files), self.data_list))
        print('')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        # if fast_proload is turned on directly load torch tensors otherwise load from fcs files
        if self.fast_preload:
            data_dict = self._load_preloaded_file(file)
        else:
            data_dict = self._load_fcs_file(file)

        # data_dict = self._transform(data_dict)
        data_dict.update({'idx': idx})

        return data_dict

    def get_filename(self, idx):
        return self.files[idx].name

    def _transform(self, data_dict):
        data = data_dict['data']
        labels = data_dict['labels']

        # Convert to torch tensors
        data = torch.from_numpy(data.values).float()
        if 'blast' in labels:
            labels = torch.from_numpy(labels['blast'].values).float()
        else:
            labels = torch.zeros(len(labels)).float()

        return {'data': data, 'labels': labels}

    def _load_preloaded_file(self, file):
        data = torch.load(str(file) + '.pt')
        labels = torch.load(str(file) + '_y.pt')

        return {'data': data, 'labels': labels}

    def _load_fcs_file(self, file):
        """
        Load fcs data from a xml file accompanied with a fcs file. the 'file' input parameter
        is the name of the directory that contains the xml file.
        """
        # Load fcs data
        with suppress_stdout_stderr():  # suppress qinfo output
            sample = flowme.fcs(str(file))

            # Remove dublicate columns if available
            events = sample.events()
            if len(set(events.columns)) != len(events.columns):
                events = events.loc[:, ~events.columns.duplicated()]

        try:
            data = events[self.marker_list]  # get the data
        except KeyError as e:
            print(f'KeyError for file {file}')
            raise e

        labels = sample.gate_labels()  # get gating (GT) information

        data_dict = {'data': data, 'labels': labels}

        return self._transform(data_dict)

    def _get_list(self):
        """
        Get list of experiment files (*.xml and *.analysis) from list self.data_list.
        """
        with open(self.data_list, 'r') as file:
            files = file.readlines()
        files = [f.strip() for f in files]
        files = [Path(f) for f in files if Path(f).is_file()]

        return files

    def _preload(self):
        """
        Preloads the dataset and stores the torch tensors directly to save loading time.
        """
        # If already loaded, set load to False
        if self.tmp_path.is_dir():
            # By storing the data as torch tensors we lose the column names. To make sure that the preloaded files
            # match to the markers we chose for the current experiment we save a text file with the marker list
            # of the preloaded file. In theory we could also just preload the file with all the markers and afterwards
            # select the correct parts of the torch tensor but just enforcing the same list of markers as is done
            # here is simpler and less error prone.
            try:
                with open(self.tmp_path / Path('info.txt'), 'r') as info_file:
                    infos = info_file.readlines()
                    infos = [x.strip() for x in infos]

                    preloaded_markers = infos[0].split(',')
            except FileNotFoundError:
                preloaded_markers = []

            # if markers do not match just preload again ...
            if preloaded_markers != self.marker_list:
                rmtree(self.tmp_path)

                print('Existing tmp folder was created with markers = {}. Preload fcs files again in {}.'
                      .format(", ".join(preloaded_markers), self.tmp_path))
                load = True
                self.tmp_path.mkdir(parents=False, exist_ok=True)
            else:
                print('Preloaded fcs files already exist in {}.'.format(self.tmp_path))
                load = False
        else:
            print('Preload fcs files in {}.'.format(self.tmp_path))
            load = True
            self.tmp_path.mkdir(parents=False, exist_ok=True)

        # Load files and save as pytorch tensor
        exp_files = self._get_list()
        file_descriptor_list = []

        for exp in tqdm(exp_files):

            # file_descriptors are here the file names of the 'data' tensor without the '.pt' ending.
            # This way the data tensor is file_descriptor + '.pt' and the label tensor is file_descriptor + '_y.pt
            file_descriptor = self.tmp_path / str(exp.stem)
            file_descriptor_list.append(file_descriptor)

            if load:
                exp_tensor_dict = self._load_fcs_file(exp)

                torch.save(exp_tensor_dict['data'],
                           str(file_descriptor) + '.pt')
                torch.save(exp_tensor_dict['labels'],
                           str(file_descriptor) + '_y.pt')

        with open(self.tmp_path / Path('info.txt'), 'w') as info_file:
            marker_str = ','.join(self.marker_list) + '\n'
            info_file.write(marker_str)

        return file_descriptor_list


class FlowDebug(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        assert kwargs['data_type'] in ['train', 'eval', 'test'], \
            'data_type must be either "train", "eval" or "test" but got {}'.format(
                kwargs['data_type'])

        self.data_type = kwargs['data_type']
        self.marker_list = kwargs['markers'].replace(' ', '').split(',')
        self.sequence_length = kwargs['sequence_length']

        print('Created {} dataset.'.format(kwargs['data_type']))

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        l1 = int(self.sequence_length/2)
        l2 = self.sequence_length - l1

        dim = len(self.marker_list)

        nfeatures = torch.zeros((l1, dim))
        nlabels = torch.zeros((l1))
        ofeatures = torch.ones((l2, dim))
        olabels = torch.ones((l2))

        data = torch.cat([nfeatures, ofeatures])
        labels = torch.cat([nlabels, olabels])

        return {'data': data, 'labels': labels}