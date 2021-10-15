from torch.optim.lr_scheduler import LambdaLR
from math import exp

def lin_exp_scheduler(optimizer, warmup_epoch=4, max_factor=100, last_epoch=-1, exponent=0.5):
    def lr_lambda(epoch):
        w = (warmup_epoch + 1.1)**(exponent + 1)
        m = max_factor * w/(warmup_epoch + 1.1)

        f1 = ((epoch + 1.1)/(w))
        f2 = 1/((epoch + 1.1)**(exponent))
        # f2 = max_factor/exp(epoch + 2)
        
        return m * min(f1, f2)

    return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)