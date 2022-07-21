import argparse
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import flowme

import flowformer
from flowformer import ConfigParser
from flowformer.utils import mrd_plot, draw_panel, draw_all, MetricTracker, tictoc
from flowformer.utils import get_project_root, suppress_stdout_stderr, read_stats

with suppress_stdout_stderr():
    import flowme



def predict(config, filename, plot_all):

    # build model architecture
    model = config.init_obj('arch', flowformer)
    print(model)

    # get function handles of loss and metrics
    # loss_fn = getattr(flowformer, config['loss'])
    metric_fns = [getattr(flowformer, met) for met in config['metrics']]
    if not 'mrd_gt' in [m.__name__ for m in metric_fns]:
        metric_fns.append(getattr(flowformer, 'mrd_gt'))
    if not 'mrd_pred' in [m.__name__ for m in metric_fns]:
        metric_fns.append(getattr(flowformer, 'mrd_pred'))
    # metrics = MetricTracker('loss', *[m.__name__ for m in metric_fns])

    marker_list = config['data_loader']['args']['markers'].replace(' ', '').split(',')

    print('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print('\n### Begin prediction ###')
    # Load data
    with suppress_stdout_stderr():  # suppress qinfo output
        sample = flowme.fcs(str(filename))

        # Remove dublicate columns if available
        events = sample.events()
        if len(set(events.columns)) != len(events.columns):
            events = events.loc[:, ~events.columns.duplicated()]

    try:
        data = events[marker_list]  # get the data
    except KeyError as e:
        print(f'marker list does not match: {filename}')
        raise e

    data = torch.from_numpy(data.values).float()

    labels = sample.gate_labels()
    if 'blast' in labels:
        labels = torch.from_numpy(labels['blast'].values).float()
    else:
        labels = torch.zeros(len(labels)).float()

    data = data.to(device).unsqueeze(0)
    target = labels.to(device).unsqueeze(0)

    with torch.no_grad():
        # Run model on data
        output = model(data)

    # Compute metrics
    for met in metric_fns:
        print(f'{met.__name__}: {met(output, target)}')


    output_folder = Path('predict') / Path(filename).stem
    output_folder.mkdir(exist_ok=True, parents=True)
        

    print('\n## Plot panel')
    fig = draw_panel(data=data, labels=target, predictions=output,
                     marker_list=marker_list, number_of_points=5000)
    fig.savefig(str(output_folder / 'panel.png'), format='png', dpi=300)

    if plot_all:
        fig_all_gt, fig_all_pred = draw_all(data=data, labels=target, predictions=output,
                        marker_list=marker_list, number_of_points=5000)
        fig_all_gt.savefig(str(output_folder / 'all_GT.png'), format='png', dpi=300)
        fig_all_pred.savefig(str(output_folder / 'all_pred.png'), format='png', dpi=300)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Apply trained model to a data sample. Computes metrics and creates plots (the top row shows the ground truth and the bottom the predictions. Blasts in red, all other cells in blue)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--file', type=str,
                      help='Input filename (either a *.xml file with corresponding fcs file or a *.analysis file)')
    args.add_argument('-p', '--plot_all', action='store_true',
                      help='Plot every marker combination. This might take around half a minute.')

    config = ConfigParser.from_args(args)
    arguments = args.parse_args()
    predict(config, arguments.file, arguments.plot_all)
