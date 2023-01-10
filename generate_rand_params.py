"""
Functions to generate random hyper-parameters
"""
from hyperopt import hp
from hyperopt.pyll.stochastic import sample

iters_per_iteration = 1


# handle floats which should be integers
# works with flat params
def handle_integers(params):

    new_params = {}
    for k, v in params.items():
        if type(v) == float and int(v) == v:
            new_params[k] = int(v)
        else:
            new_params[k] = v

    return new_params


space = {
    'learning_rate': hp.uniform('learning_rate', 9e-6, 2e-3),
    'weight_decay': hp.uniform('weight_decay', 9e-5, 3e-3),
}


def get_params(args):

    params = sample(space)
    params = handle_integers(params)

    params['batchsize'] = args.batchsize  # i just need Batchsize from args

    return params
