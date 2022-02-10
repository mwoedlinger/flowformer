import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict, Mapping
from time import time
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def read_stats(filename):
    with open(str(filename), 'r') as file:
        stats_str = json.load(file)

    markers = list(stats_str['mean'].keys())
    mean = {m: float(stats_str['mean'][m]) for m in markers}
    std = {m: float(stats_str['std'][m]) for m in markers}
    stats = {'mean': mean, 'std': std}

    return stats


def tictoc():
    """
    Returns time in seconds since the last time the function was called.
    For the initial call 0 is returned.
    """
    toc = time()

    try:
        dt = toc - tictoc.tic
    except AttributeError:
        dt = 0

    tictoc.tic = toc
    
    return dt


def ptictoc(name=None):
    """
    Returns and prints (!) time in seconds since the last time the function was called.
    For the initial call 0 is returned.
    """
    if not hasattr(tictoc, 'tic'):
        tictoc.tic = time()
        initial = True
    else:
        initial = False

    toc = time()
    dt = toc - tictoc.tic
    tictoc.tic = toc

    if not initial:
        if name is None:
            print('dt = {}'.format(dt))
        else:
            print('dt_{} = {}'.format(name, dt))

    return dt

# Define a context manager to suppress stdout and stderr.
# Taken from: https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]

        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)

        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def read_config_json(fname):
    """
    Load config from json file and perform consistency checks
    Performs consistency checks of the config file for the reformer model and flow data.
    """

    fname = Path(fname)
    with fname.open('rt') as handle:
        config = json.load(handle, object_hook=OrderedDict)

    # Check validity of config
    c = config['arch']['args']
    c_model_name = config['arch']['type']
    if c_model_name == 'FlowTransformer':
        # c_data = config['data_loader']['args']
        # assert c['axial_pos_shape'][0]*c['axial_pos_shape'][1] == c_data['sequence_length'], \
        #     'sequence_length must be axial_pos_shape_list**2'
        assert len(c['conv1d_decoder']) == len(c['conv1d_kernel']), \
            'length of decoder parameters must be equal to length of kernel parameters'

    markers = config['data_loader']['args']['markers'].replace(
        ' ', '').split(',')
    config['arch']['args']['_num_markers'] = len(markers)
    config['arch']['args']['_sequence_length'] = config['data_loader']['args']['sequence_length']

    return config


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = {key: [] for key in keys}
        self.reset()

    def reset(self):
        for key in self._data.keys():
            self._data[key] = []

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data[key] += [value] * n

    def avg(self, key):
        return np.mean(self._data[key])

    def median(self, key):
        return np.median(self._data[key])

    def data(self):
        return self._data

    def result(self):
        avg_dict = {key: self.avg(key) for key in self._data.keys()}
        median_dict = {key: self.median(key) for key in self._data.keys()}
        return avg_dict, median_dict


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent


def dict_merge(dct, merge_dct, verify=False):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :param verify: checks if no entry is added to the dictionary
    :return: None
    """
    #     dct = copy.copy(dct)
    changes_values = {}
    changes_lists = {}

    for k, _ in merge_dct.items():
        if verify:
            assert k in dct, 'key "{}" is not part of the default dict'.format(
                k)
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], Mapping)):
            changes_lists[k] = dict_merge(dct[k], merge_dct[k], verify=verify)
        else:
            if k in dct and dct[k] != merge_dct[k]:
                changes_values[k] = merge_dct[k]

            dct[k] = merge_dct[k]

    _sorted = []
    for k, _ in dct.items():
        if k in changes_values:
            _sorted.append((k, changes_values[k]))
        elif k in changes_lists:
            _sorted.extend(changes_lists[k])

    return _sorted


def draw_cells(data, labels, predictions, markers=[1, 2], marker_names=['FSC-A', 'FSC-W'], number_of_points=10000):
    """
    Creates a scatter plot where x and y axes are given by the markers with index markers[0] and markers[1]
    """

    if len(predictions.shape) > 2:
        pred_labels = torch.argmax(predictions, dim=1)
    elif len(predictions.shape) == 2:
        pred_labels = 1.0*(predictions > 0.5)
    else:
        raise NotImplementedError

    x_data = data[0, :number_of_points, markers[0]].cpu().detach().numpy()
    y_data = data[0, :number_of_points, markers[1]].cpu().detach().numpy()
    colors_GT = ['red' if x > 0.5 else 'blue' for x in labels[0,
                                                              :number_of_points].cpu().detach().numpy()]
    colors_pred = ['red' if x > 0.5 else 'blue' for x in pred_labels[0,
                                                                     :number_of_points].cpu().detach().numpy()]

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    marker_size_GT = [2 if colors_GT[n] ==
                      'red' else 1 for n, m in enumerate(colors_GT)]
    marker_sizepred = [2 if colors_GT[n] ==
                       'red' else 1 for n, m in enumerate(colors_pred)]
    ax[0].scatter(x_data, y_data, s=marker_size_GT, c=colors_GT)
    ax[0].set_xlabel(marker_names[0])
    ax[0].set_ylabel(marker_names[1])
    ax[0].title.set_text('GT')

    ax[1].scatter(x_data, y_data, s=marker_sizepred, c=colors_pred)
    ax[1].set_xlabel(marker_names[0])
    ax[1].set_ylabel(marker_names[1])
    ax[1].title.set_text('pred')

    return fig

def draw_single_projection(data: pd.DataFrame, labels: np.array, x: str='CD10', y: str='CD45', num_samples=10000, **kwargs):
    d = data[:num_samples]
    p = (labels[:num_samples] > 0.5)*1
    
    fig = px.scatter(d,
                     x=x, y=y,
                     color=p,
                     **kwargs)
    return fig
    
def draw_projections(data: pd.DataFrame, labels: np.array, predictions: np.array, marker_pairs: list=[['CD45', 'CD10']], num_samples=1000, **kwargs):
    d = data[10000:10000+num_samples]
    p = (predictions[10000:10000+num_samples] > 0.5)*1
    l = labels[10000:10000+num_samples]
    num_plots = len(marker_pairs)
        
    fig = make_subplots(rows=1, cols=num_plots)
    fig.update_layout(template='simple_white')
    
    # Color the predictions according to correctness
    tn = np.logical_not(np.logical_or(l, p))    # healthy
    tp = np.logical_and(l, p)                   # blast
    fp = np.logical_and(np.logical_not(l), p)   # blast predicted but healthy
    fn = np.logical_and(l, np.logical_not(p))   # healthy predicted but blast

    cdict = {
        1: 'rgb(83,133,251)',
        2: 'rgb(187,0,4)',
        3: 'rgb(196,79,240)',
        4: 'rgb(111,230,89)'
    }
    colors = [cdict[c] for c in (tn + 2*tp + 3*fp + 4*fn)]    

    for n, m in enumerate(marker_pairs):
        fig.add_trace(
            go.Scatter(x=d[m[0]], y=d[m[1]], 
                       mode='markers', 
                       showlegend=False,
                       marker={
                           'color': colors, 
                           'size': 4}),
            row=1, col=n+1
        )
        y_max = max(d[m[1]])
        if n == 0:
            fig.update_layout(
                **{'xaxis': {'title': {'text': m[0]}},
                   'yaxis': {'title': {'text': m[1]},
                             'range': [0,y_max+0.1]}}
            )
        else:
            fig.update_layout(
                **{'xaxis'+str(n+1): {'title': {'text': m[0]}},
                   'yaxis'+str(n+1): {'title': {'text': m[1]},
                             'range': [0,y_max+0.1]}}
            )

    fig.update_layout(**kwargs)
    
    return fig



def draw_all(data, labels, predictions, marker_list, number_of_points=5000):
    """
    Creates a grid of scatter plots for all possible marker combinations.
    """

    if len(predictions.shape) > 2:
        pred_labels = torch.argmax(predictions, dim=1)
    elif len(predictions.shape) == 2:
        pred_labels = 1.0*(predictions > 0.5)
    else:
        raise NotImplementedError

    min_fig_size = 5
    num_markers = data.shape[2]

    fig_gt, ax_gt = plt.subplots(num_markers, num_markers, figsize=(
        min_fig_size*num_markers, min_fig_size*num_markers))
    fig_pred, ax_pred = plt.subplots(num_markers, num_markers, figsize=(
        min_fig_size*num_markers, min_fig_size*num_markers))

    for m0 in range(num_markers):
        for m1 in range(num_markers):

            x_data = data[0, :number_of_points, m0].cpu().detach().numpy()
            y_data = data[0, :number_of_points, m1].cpu().detach().numpy()
            colors_GT = ['red' if x > 0.5 else 'blue' for x in labels[0,
                                                                      :number_of_points].cpu().detach().numpy()]
            colors_pred = ['red' if x > 0.5 else 'blue' for x in pred_labels[0,
                                                                             :number_of_points].cpu().detach().numpy()]

            marker_size_GT = [2 if colors_GT[n] ==
                              'red' else 1 for n, m in enumerate(colors_GT)]
            marker_sizepred = [2 if colors_GT[n] ==
                               'red' else 1 for n, m in enumerate(colors_pred)]
            ax_gt[m0, m1].scatter(
                x_data, y_data, s=marker_size_GT, c=colors_GT)
            ax_gt[m0, m1].set_xlabel(marker_list[m0])
            ax_gt[m0, m1].set_ylabel(marker_list[m1])
            #ax_gt[m0, m1].title.set_text('GT')

            ax_pred[m0, m1].scatter(
                x_data, y_data, s=marker_sizepred, c=colors_pred)
            ax_pred[m0, m1].set_xlabel(marker_list[m0])
            ax_pred[m0, m1].set_ylabel(marker_list[m1])
            #ax_pred[m0, m1].title.set_text('pred')

    return fig_gt, fig_pred


def draw_panel(data, labels, predictions, marker_list, number_of_points=10000):
    """
    Creates a scatter plot for the marker combinations in the standard panel.
    """

    if len(predictions.shape) > 2:
        pred_labels = torch.argmax(predictions, dim=1)
    elif len(predictions.shape) == 2:
        pred_labels = 1.0*(predictions > 0.5)
    else:
        raise NotImplementedError

    panels = [
        ['FSC-A', 'SSC-A'],
        ['CD45', 'SSC-A'],
        ['CD19', 'SSC-A'],
        ['CD10', 'SSC-A'],
        #['CD58', 'CD10'],
        ['CD45', 'CD10'],
        ['CD20', 'CD10'],
        ['CD34', 'CD10'],
        ['CD38', 'CD10'],
        ['CD10', 'CD19']
    ]

    min_fig_size = 6
    num_plots = len(panels)

    fig, ax = plt.subplots(2, num_plots, figsize=(
        min_fig_size*num_plots, min_fig_size*2))

    for p in range(num_plots):
        m0_str = panels[p][0]
        m0 = marker_list.index(m0_str)
        m1_str = panels[p][1]
        m1 = marker_list.index(m1_str)

        x_data = data[0, :number_of_points, m0].cpu().detach().numpy()
        y_data = data[0, :number_of_points, m1].cpu().detach().numpy()
        colors_GT = ['red' if x > 0.5 else 'blue' for x in labels[0,
                                                                  :number_of_points].cpu().detach().numpy()]
        colors_pred = ['red' if x > 0.5 else 'blue' for x in pred_labels[0,
                                                                         :number_of_points].cpu().detach().numpy()]

        marker_size_GT = [1.5 if colors_GT[n] ==
                          'red' else 0.75 for n, m in enumerate(colors_GT)]
        marker_sizepred = [1.5 if colors_GT[n] ==
                           'red' else 0.75 for n, m in enumerate(colors_pred)]
        ax[0, p].scatter(x_data, y_data, s=marker_size_GT, c=colors_GT)
        ax[0, p].set_xlabel(m0_str)
        ax[0, p].set_ylabel(m1_str)
        # if p == 0:
        #     ax[0, p].title.set_text('GT')

        ax[1, p].scatter(x_data, y_data, s=marker_sizepred, c=colors_pred)
        ax[1, p].set_xlabel(m0_str)
        ax[1, p].set_ylabel(m1_str)
        # if p == 0:
        #     ax[1, p].title.set_text('pred')

    return fig


def mrd_plot(mrd_list_gt, mrd_list_pred, f1_score, filenames=None):
    """
    Creates a plot for true and predicted mrd values.
    """

    if filenames is None:
        filenames = [str(n) for n in range(len(mrd_list_gt))]

    min_val = 1.0e-5

    # Set min mrd to min_val:
    mrd_list_gt = [mrd if mrd >= min_val else min_val for mrd in mrd_list_gt]
    mrd_list_pred = [mrd if mrd >= min_val else min_val for mrd in mrd_list_pred]

    # Create figure
    data = pd.DataFrame(list(zip(mrd_list_gt, mrd_list_pred, f1_score, filenames)), 
                        columns=['gt', 'pred', 'f1_score', 'names'])
    fig = px.scatter(data,
                     x='gt', y='pred', 
                     log_x=True, log_y=True, range_x=[min_val, 1], range_y=[min_val, 1],
                     color='f1_score', template='simple_white', hover_name='names')
    
    # Add diagonal line
    fig.add_shape(type='line',
        x0=0, y0=0, x1=1, y1=1,
        line=dict(
            color='Gray',
            width=2,
            dash='dashdot',
        )
    )
    
    # Add vertical line
    fig.add_shape(type='line',
        x0=5.0e-4, y0=0, x1=5.0e-4, y1=1,
        line=dict(
            color='Gray',
            width=2,
            dash='dot',
        )
    )
    
    # Add horizontal line
    fig.add_shape(type='line',
        x0=0, y0=5.0e-4, x1=1, y1=5.0e-4,
        line=dict(
            color='Gray',
            width=2,
            dash='dot',
        )
    )
    return fig

# def mrd_plot(mrd_list_gt, mrd_list_pred, f1_score):
#     """
#     Creates a plot for true and predicted mrd values.
#     """

#     f1_score = np.array(f1_score)
#     colors = (np.outer((1-f1_score), [1.0, 0.0, 0.0]
#                        ) + np.outer(f1_score, [0.0, 1.0, 0.0]))
#     offset_frac = 3
#     hline = 5.0e-4
#     vline = 5.0e-4
#     offset = 1-1/offset_frac
#     min_val = 1.0e-5

#     # Set min mrd to min_val:
#     mrd_list_gt = [mrd if mrd >= min_val else min_val for mrd in mrd_list_gt]
#     mrd_list_pred = [mrd if mrd >=
#                      min_val else min_val for mrd in mrd_list_pred]

#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Helper lines
#     ax.plot([0, 1], [0, 1], c='black', linestyle='-')
#     ax.plot([0, 1], [hline, hline], c='grey', linestyle='--')
#     ax.plot([vline, vline], [0, 1], c='grey', linestyle='--')
#     ax.plot([0, 1], [0, 1-offset], c='lightblue', linestyle='--')
#     ax.plot([0, 1-offset], [0, 1], c='lightblue', linestyle='--')

#     # Draw points
#     ax.scatter(mrd_list_pred, mrd_list_gt, s=10, c=colors)

#     # Adjust graph settings
#     ax.set_ylim(min_val, 1)
#     ax.set_xlim(min_val, 1)
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.set_ylabel('True')
#     ax.set_xlabel('Predicted')

#     return fig


def mrd_lineplot(mrd_list_gt, mrd_list_pred, f1_score):
    idx = sorted(range(len(mrd_list_gt)), key=lambda k: mrd_list_gt[k])

    vline1 = sum(mrd_list_gt[idx] <= 0.0001)
    vline2 = sum(mrd_list_gt[idx] <= 0.0005)
    vline3 = sum(mrd_list_gt[idx] <= 0.001)

    f1_sorted = f1_score[idx]
    mrd_gt_sorted = mrd_list_gt[idx]
    mrd_pred_sorted = mrd_list_pred[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    X = np.linspace(0, len(idx), len(idx))

    ax.scatter(X, f1_sorted, s=5, c='maroon')  # c='skyblue')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('f1 score')
    ax.legend(['f1 scores'], loc='upper left', framealpha=1.0)
    ax.set_yticks(ticks=[0, 0.25, 0.5, 0.75, 1.0])
    ax.axvline(vline1, linewidth='1.0', c='indigo')
    ax.text(vline1+2, 1.01, 'MRD < 0.01%', rotation=30, c='indigo')
    ax.axvline(vline2, linewidth='1.0', c='indigo')
    ax.text(vline2+2, 1.01, 'MRD < 0.05%', rotation=30, c='indigo')
    ax.axvline(vline3, linewidth='1.0', c='indigo')
    ax.text(vline3+5, 1.01, 'MRD < 0.1%', rotation=30, c='indigo')
    ax.set_ylim(bottom=0, top=1.0)
    ax.set_xlim(left=0, right=200)

    ax2 = ax.twinx()

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color('indigo')
    ax2.plot(X, mrd_gt_sorted, c='indigo')
    ax2.plot(X, mrd_pred_sorted, c='seagreen')
    ax2.set_yscale('log')
    ax2.legend(['mrd GT', 'MRD Pred'], loc='lower right')
    ax2.set_ylabel('MRD')
    ax2.set_ylim(bottom=mrd_gt_sorted[0], top=1.0)

    return fig
