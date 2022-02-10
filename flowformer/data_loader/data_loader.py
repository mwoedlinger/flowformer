import torch
from .dataset import FlowData, FlowDebug


class FlowDataLoader(torch.utils.data.DataLoader):
    """
    Dataloader for flowdata. Basically the standard pytorch dataloader, the action happens in the FlowData
    Dataset class.
    """

    def __init__(self, **kwargs):
        dataset = FlowData(**kwargs)

        if kwargs['data_type'] != 'train':
            kwargs['batch_size'] = 1

        super().__init__(dataset=dataset, batch_size=kwargs['batch_size'], num_workers=kwargs['num_workers'])


class FlowDebugLoader(torch.utils.data.DataLoader):
    def __init__(self, **kwargs):
        dataset = FlowDebug(**kwargs)

        if kwargs['data_type'] != 'train':
            kwargs['batch_size'] = 1
            kwargs['shuffle'] = False

        super().__init__(dataset=dataset, batch_size=kwargs['batch_size'], shuffle=kwargs['shuffle'],
                         num_workers=kwargs['num_workers'])
