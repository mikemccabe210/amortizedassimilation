import torch

def filter_obs(filt):
    """ Observes the specified coordinates """
    def inner(x):
        if len(x.shape) == 4:
            return x[:, :, :, filt]
        elif len(x.shape) == 3:
            return x[:, :, filt]
        else:
            return x[:, filt]
    return inner

def mystery_operator():
    """ Creates a random projection matrix for
    random lossy feature generation. """
    proj = torch.randn(40, 10)
    def inner(x):
        return x @ proj
    return inner