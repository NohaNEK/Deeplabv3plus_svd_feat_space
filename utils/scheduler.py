from torch.optim.lr_scheduler import _LRScheduler, StepLR

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
        print(self.min_lr )
        print(self.max_iters )
        print(self.power )
        
    
    def get_lr(self):
        # print("last epoch",self.last_epoch)
        # print("base lrs",self.base_lrs)
        # print(( 1 - self.last_epoch/self.max_iters ))
        # print(( 1 - self.last_epoch/self.max_iters )**self.power)
        # print("power0.7 ",( 1 - self.last_epoch/self.max_iters )**(0.7))
        
        # print("power0.5 ",( 1 - self.last_epoch/self.max_iters )**(0.5))
        # print("power0.2 ",( 1 - self.last_epoch/self.max_iters )**(0.2))
        # print("power0.009 ",( 1 - self.last_epoch/self.max_iters )**(0.009))
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]