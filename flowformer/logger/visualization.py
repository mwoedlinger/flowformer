import importlib
from datetime import datetime

class WandbWriter():
    def __init__(self, log_dir, logger, enabled, model):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            try:
                self.wandb = importlib.import_module('wandb')
            except ImportError:
                message = "Warning: visualization (wandb) is configured to use, but currently not installed on " \
                          "this machine. Please install wandb with 'pip install wandb' or turn off the option in the 'config.json' file."
                logger.warning(message)

        # TODO: specify wandb run with name, config, etc.
        self.wandb.init(project='transformerflow')
        self.wandb.watch(model, log='all', log_freq=10)

        self._step = 0
        self.mode = ''
        self.logs = {}

        self.timer = datetime.now()

    def set_mode(self, mode='train'):
        self.mode = mode

    def step(self):
        self._step += 1
        
        duration = datetime.now() - self.timer
        self.log('steps_per_sec', 1 / duration.total_seconds())
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self._step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.log('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def add_scalar(self, key, val):
        self.log(key, val)

    def add_figure(self, key, val):
        # self.log(key, self.wandb.Image(val)) # This would log the plot as an image
        self.log(key, val)

    def log(self, tag, val):
        logs = {f'{self.mode}/{tag}': val}
        self.logs.update(logs)

    def commit(self):
        self.wandb.log(self.logs, step=self._step)
        self.logs = {}

class TensorboardWriter():
    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding', 'add_figure'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr
