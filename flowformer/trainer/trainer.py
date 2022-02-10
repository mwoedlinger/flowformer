import numpy as np
import torch
from torchvision.utils import make_grid
from tqdm import tqdm
import pandas as pd
from ..base import BaseTrainer
from ..utils import inf_loop, MetricTracker, draw_projections, mrd_plot
from ..utils import ptictoc as pt


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None):

        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config

        # Data
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader

        self.len_epoch = len(self.data_loader)
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # Metrics for training and validation
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.writer.set_mode('train')
        self.train_metrics.reset()
        for batch_idx, batch in tqdm(enumerate(self.data_loader), desc='train', total=len(self.data_loader)):
            # Load data
            data = batch['data'].to(self.device)
            target = batch['labels'].to(self.device)

            # Forward + backward pass
            self.optimizer.zero_grad()
            output = self.model(data) # Apply model
            
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.50)
            self.optimizer.step()

            # Write metrics etc. to tensorboard
            self.writer.step()
            # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, mode='train')
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

                # Draw prediction vs GT:
                vis_markers = self.config['trainer']['visualize']
                marker_list = self.config['data_loader']['args']['markers'].replace(' ', '').split(',')

                data = pd.DataFrame(data[0].cpu().numpy(), columns=marker_list)
                self.writer.add_figure('default_panel', 
                                        draw_projections(data=data, labels=target[0].cpu().numpy(), predictions=output[0].detach().cpu().numpy(),
                                                        marker_pairs=vis_markers, 
                                                        height=400, width=1600))

            if batch_idx == self.len_epoch:
                break
            
            self.writer.commit()

        log, log_median = self.train_metrics.result()

        if self.do_validation:
            val_log, val_log_median = self._valid_epoch(epoch)   
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log.update(**{'val_median_f1_score' : val_log_median['f1_score']})                                                     
            self.writer.commit()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.writer.set_mode('eval')
        self.valid_metrics.reset()
        filenames = []
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(self.valid_data_loader), desc= 'eval', total=len(self.valid_data_loader)):
                # Load data
                data = batch['data'].to(self.device)
                target = batch['labels'].to(self.device)
                filenames.append(self.valid_data_loader.dataset.get_filename(batch['idx']))

                # Forward pass
                output = self.model(data) # Apply model
                loss = self.criterion(output, target)

                # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'eval')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                # self.writer.add_figure('input', make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx % self.log_step == 0:
                    # Draw prediction vs GT:
                    vis_markers = self.config['trainer']['visualize']
                    marker_list = self.config['data_loader']['args']['markers'].replace(' ', '').split(',')

                    data = pd.DataFrame(data[0].cpu().numpy(), columns=marker_list)
                    self.writer.add_figure('default_panel', 
                                            draw_projections(data=data, labels=target[0].cpu().numpy(), predictions=output[0].detach().cpu().numpy(),
                                                            marker_pairs=vis_markers, 
                                                            height=400, width=1600))

        # MRD figure
        metric_data = self.valid_metrics.data()
        mrd_fig = mrd_plot(mrd_list_gt=metric_data['mrd_gt'], mrd_list_pred=metric_data['mrd_pred'], f1_score=metric_data['f1_score'], filenames=filenames)
        self.writer.add_figure('MRD', mrd_fig)

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
