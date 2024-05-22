class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.accumulator = 0
        self.n = 0

    def update(self, value, nums):
        self.accumulator += value
        self.n += nums

    def average(self):
        ret = self.accumulator / self.n
        self.reset()
        return ret

    def reset(self):
        self.accumulator = 0
        self.n = 0
