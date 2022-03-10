import argparse
import collections
import torch
import numpy as np

import flowformer
#from flowformer import LogTerminal
from flowformer import ConfigParser
from flowformer import Trainer


# fix random seeds for reproducibility
SEED = 41
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def train(config_parser):
    print('\n### Prepare train data ###')
    logger = config_parser.get_logger('train')

    # setup data_loader instances
    data_loader = config_parser.init_obj(
        'data_loader', flowformer, data_type='train')
    valid_data_loader = config_parser.init_obj(
        'data_loader', flowformer, data_type='eval')

    # build model architecture, then print to console
    model = config_parser.init_obj('arch', flowformer)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(flowformer, config_parser['loss'])
    metrics = [getattr(flowformer, met)
               for met in config_parser['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config_parser.init_obj(
        'optimizer', torch.optim, trainable_params)

    lr_scheduler = config_parser.init_obj(
        'lr_scheduler', [torch.optim.lr_scheduler, flowformer], optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config_parser,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    print('\n### Begin training ###')
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Train model on gated data.')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'],
                   type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target='data_loader;args;batch_size')
    ]
    config_parser = ConfigParser.from_args(args, options)
    train(config_parser)
