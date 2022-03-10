import argparse
import torch
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import flowme

import flowformer
from flowformer import ConfigParser
from flowformer.model.metric import *
from flowformer.utils import suppress_stdout_stderr, draw_single_projection


def load_fcs_file(file, marker_list):
    """
    Load fcs data from a xml file accompanied with a fcs file. the 'file' input parameter
    is the name of the directory that contains the xml file.
    """
    # Load fcs data
    with suppress_stdout_stderr():  # suppress qinfo output
        sample = flowme.fcs(str(file))

        # Remove dublicate columns if available
        events = sample.events()
        print(events.columns)
        if len(set(events.columns)) != len(events.columns):
            events = events.loc[:, ~events.columns.duplicated()]

    try:
        data = events[marker_list]  # get the data
    except KeyError as e:
        print(f'KeyError for file {file}')
        raise e

    return torch.from_numpy(data.values).float()

def predict(config, filename, output_dir):
    # build model architecture
    model = config.init_obj('arch', flowformer)

    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    marker_list = config['data_loader']['args']['markers'].replace(' ', '').split(',')
    
    with torch.no_grad():

        print(f'## Load {filename}')
        sample = load_fcs_file(filename, marker_list).to(device).unsqueeze(0)

        prediction = model(sample)
        labels = 1*(prediction[0]>0.5).cpu().numpy()

        print('## Creating plots...')
        for m1 in tqdm(marker_list):
            for m2 in marker_list:
                if m1 == m2:
                    continue

                data = pd.DataFrame(sample[0].cpu().numpy(), columns=marker_list)
                fig = draw_single_projection(data=data, labels=labels, x=m1, y=m2, num_samples=5000)
                fig.write_image(Path(output_dir) / (Path(filename).stem+'_'+m1+'_'+m2+'.png'))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Create prediction for a specified sample.')
    args.add_argument('argv', nargs=1, help='filename of fcs file')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-o', '--output_dir', default='.', type=str,
                      help='directory to store output (default: . ')

    config = ConfigParser.from_args(args)
    predict(config, args.parse_args().argv[0], args.parse_args().output_dir)
