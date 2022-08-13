import os
import warnings
import numpy as np
import scipy.io
from copy import deepcopy
from pandas import read_csv

from qrnet.controllers import load_NN
from qrnet.simulate import monte_carlo
from qrnet.validate import find_fixed_point

from run.utilities import yn_input, make_parser

from problem import OCP, config, model_dir, data_dir, results_dir

parser = make_parser(mt=True, rs=True, xd=True, io=True, o=True, v=True)
parser.add_argument(
    '-r', '--reverse_order', dest='reverse_order',
    action='store_true',
    help='If running multiple timestamps start from most recent'
)
args = parser.parse_args().__dict__

X0_pool = None

import_open_loop = args.pop('import_open_loop')

if import_open_loop:
    test_data = scipy.io.loadmat(os.path.join(data_dir, 'test.mat'))

    n_trajectories = int(test_data['n_trajectories'])
    n_MC = config.n_trajectories_MC
    if n_trajectories != n_MC:
        raise ValueError('Test data must have %d trajectories.' % n_MC)

    seed = np.unique(test_data['seed'])
    if len(seed) != 1 or np.isnan(seed):
        raise ValueError('Test data seed(s) is not valid: ', seed)

    if args['X0_distance'] is not None:
        raise ValueError('Cannot specify X0_distance if importing test data.')

    args['random_seed'] = int(seed)
    args['solve_open_loop'] = False

    opt_costs = np.empty(n_trajectories)
    opt_final_times = np.empty(n_trajectories)
    X0_pool = np.empty((OCP.n_states, n_trajectories))

    t0_idx = np.where(test_data['t'] == 0.)[1]
    t0_idx = np.append(t0_idx, test_data['t'].shape[1])

    for i in range(n_trajectories):
        k0 = t0_idx[i]
        k1 = t0_idx[i+1]

        opt_costs[i] = test_data['V'][0,k0]
        opt_final_times[i] = test_data['t'][0,k1-1]
        X0_pool[:,i] = test_data['X'][:,k0]

# If a specific timestamp was supplied, test that model
# Otherwise, run through all models which haven't yet been tested
timestamp = args.pop('timestamp')

if timestamp is not None:
    timestamps = [timestamp]
else:
    info_path = os.path.join(model_dir, 'model_info.csv')
    timestamps = read_csv(info_path, usecols=['timestamp'])
    timestamps = timestamps.values.flatten().astype(int)

test_results_dir = os.path.join(results_dir, 'monte_carlo')
os.makedirs(test_results_dir, exist_ok=True)
test_results_list = os.listdir(test_results_dir)

if len(timestamps) > 1:
    # If didn't specify a timestamp, filter the list down to those which do not
    # yet have MC test results
    timestamps = [
        timestamp for timestamp in np.sort(timestamps)
        if 'test_' + str(timestamp) + '.mat' not in test_results_list
    ]

if args.pop('reverse_order'):
    timestamps = timestamps[::-1]

s1 = '\nRunning Monte Carlo simulations for model {n:d}/{N:d}...'
s2 = '({a:s}, timestamp {mt:d})\n'

for n, timestamp in enumerate(timestamps):
    controller, timestamp = load_NN(model_dir, timestamp, verbose=False)

    print(s1.format(n=n+1, N=len(timestamps)))
    print(s2.format(a=controller.architecture(), mt=timestamp))

    results_dict = monte_carlo(
        OCP, config, controller, X0_pool=X0_pool,
        suppress_warnings=~args['verbose'], **args
    )

    if import_open_loop:
        results_dict['opt_final_times'] = opt_final_times
        results_dict['opt_costs'] = opt_costs
        results_dict['ocp_converged'] = np.ones(n_trajectories, dtype=bool)

    test_results_path = os.path.join(
        test_results_dir, 'test_' + str(timestamp) + '.mat'
    )

    try:
        old_results_dict = scipy.io.loadmat(test_results_path)

        overwrite_data = yn_input('Overwrite existing test data?')

        if overwrite_data:
            if yn_input('Are you sure you want to overwrite?'):
                raise FileNotFoundError

        for key in results_dict:
            results_dict[key] = np.hstack((
                old_results_dict[key], np.reshape(results_dict[key], (1,-1))
            ))
    except FileNotFoundError:
        pass

    scipy.io.savemat(test_results_path, results_dict)

    if all([
            len(timestamps) > 1,
            args['solve_open_loop'],
            results_dict['ocp_converged'].all()
        ]):
        print('All OCPs converged, no longer attempting to solve OCPs')
        args['solve_open_loop'] = False
