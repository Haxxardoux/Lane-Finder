class Params(object):
    """
    Params : batch_size, epochs, lr
    """
    def __init__(self, batch_size, epochs, lr):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr