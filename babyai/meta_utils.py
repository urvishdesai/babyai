import torch


def get_params(module, memo=None, pointers=None):
    """ Returns an iterator over PyTorch module parameters that allows to update parameters
        (and not only the data).
    ! Side effect: update shared parameters to point to the first yield instance
        (i.e. you can update shared parameters and keep them shared)
    Yields:
        (Module, string, Parameter): Tuple containing the parameter's module, name and pointer
    """
    if memo is None:
        memo = set()
        pointers = {}
    for name, p in module._parameters.items():
        if p not in memo:
            memo.add(p)
            pointers[p] = (module, name)
            yield module, name, p
        elif p is not None:
            prev_module, prev_name = pointers[p]
            module._parameters[name] = prev_module._parameters[prev_name] # update shared parameter pointer
    for child_module in module.children():
        for m, n, p in get_params(child_module, memo, pointers):
            yield m, n, p


