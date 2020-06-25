class Params(object):
    """
    Params : batch_size, epochs, lr
    """
    def __init__(self, batch_size, epochs, lr):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr


def count_parameters(model):
    """
    Counts the total number of parameters in a model
    Args:
        model (Module): Pytorch model, the total number of parameters for this model will be counted. 

    Returns: Int, number of parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device_from_model(model):
    if hasattr(model, "weight"):
        return model.weight.device
    else:
        return get_device_from_model(list(model.children())[0])