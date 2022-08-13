_model_registry = {}

def register(model_name, model_class):
    '''
    Add a model class to the factory dictionary.

    Parameters
    ----------
    model_name : str
        Key to associate with model_class
    model_class : class reference
        Class definition to add
    '''
    _model_registry[model_name] = model_class

def available_models():
    """
    Returns a list of currently registered NN names.
    """
    return list(_model_registry.keys())

def get_model_class(model_name):
    '''
    Get a model class by name lookup.

    Arguments
    ----------
    model_name : str
        Name of the model class reference to look up

    Returns
    ----------
    model_class : class reference
        Class definition associated with model_name
    '''
    model_class = _model_registry.get(model_name)

    if model_class is None:
        err_str = model_name + ' is not a registered model.'
        err_str += ' Available model names are:\n'
        err_str += '\n'.join(_model_registry.keys())
        raise ValueError(err_str)

    return model_class
