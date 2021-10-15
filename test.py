import argparse
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

import flowformer
from flowformer import ConfigParser
from flowformer.utils import mrd_plot, draw_panel, MetricTracker, tictoc


# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def test(config):


    print('\n### Prepare test data ###')
    logger = config.get_logger('test')

    # setup data_loader instances
    test_data_loader = config.init_obj(
        'data_loader', flowformer, data_type='test')
    test_dataset = test_data_loader.dataset

    # build model architecture
    model = config.init_obj('arch', flowformer)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(flowformer, config['loss'])
    metric_fns = [getattr(flowformer, met) for met in config['metrics']]
    if not 'mrd_gt' in [m.__name__ for m in metric_fns]:
        metric_fns.append(getattr(flowformer, 'mrd_gt'))
    if not 'mrd_pred' in [m.__name__ for m in metric_fns]:
        metric_fns.append(getattr(flowformer, 'mrd_pred'))
    metrics = MetricTracker('loss', *[m.__name__ for m in metric_fns])

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    file_list = []
    data_len = len(test_dataset)

    print(f'\n## Create output folder {config["name"]}')
    out_folder = Path('output') / Path(config['name'])
    out_folder.mkdir(parents=True)

    print('\n### Begin testing ###')
    with torch.no_grad():

        for i in tqdm(range(data_len)):
            batch = test_dataset[i]
            data = batch['data'].to(device).unsqueeze(0)
            target = batch['labels'].to(device).unsqueeze(0)
            filename = str(test_dataset.files[i].name)
            file_list.append(filename)

            output = model(data)

            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size

            metrics.update('loss', loss.item())
            for met in metric_fns:
                metrics.update(met.__name__, met(output, target))

    # Log results for every file
    log_list = []
    metric_names_list = [f'{f.__name__:<{25}}' for f in metric_fns]
    metric_data = metrics.data()

    log_list.append(f'\n{"filename:":{25}}' + ' '.join(metric_names_list))
    for n, file in enumerate(file_list):
        metric_result_list = [
            f'{metric_data[m.__name__][n]:<{25}.{5}}' for m in metric_fns]
        log_list.append(f'{file:{25}}' + ' '.join(metric_result_list))

    n_samples = len(test_data_loader.sampler)
    results = {'loss': total_loss / n_samples}
    metric_results_mean, metric_results_median = metrics.result()
    results.update(metric_results_mean)
    results.update(
        **{'median_'+k: v for k, v in metric_results_median.items()})
    log_list.append(' ')
    log_list += [f'{key:{25}}: {results[key]:<{25}.{5}}' for key in results.keys()]

    with open(str(out_folder / 'test_out.txt'), 'w') as text_file:
        for l in log_list:
            logger.info(l)
            text_file.write(l+'\n')

    print('\n## Create MRD plot')
    fig = mrd_plot(mrd_list_gt=metric_data['mrd_gt'],
                   mrd_list_pred=metric_data['mrd_pred'], f1_score=metric_data['f1_score'])
    fig.savefig(str(out_folder / 'out_mrd.pdf'), format='pdf')

    print('\n## Plot panel')
    marker_list = config['data_loader']['args']['markers'].replace(
        ' ', '').split(',')
    fig = draw_panel(data=data, labels=target, predictions=output,
                     marker_list=marker_list, number_of_points=5000)
    fig.savefig(str(out_folder / 'panel.png'), format='png', dpi=300)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    test(config)
