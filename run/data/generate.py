import os
import numpy as np
import scipy.io

from qrnet.generate import generate
from qrnet.controllers import load_NN

from run.utilities import make_parser, yn_input

from problem import OCP, config, model_dir, data_dir

parser = make_parser(rs=True, dt=True, mt=True, v=True)
args = parser.parse_args()

if args.timestamp:
    # Loads pre-trained controller for NN warm start
    controller, _ = load_NN(model_dir, args.timestamp)
else:
    # LQR warm start
    controller = OCP.LQR

if args.data_type == 'train':
    n_trajectories = config.n_trajectories_train
elif args.data_type == 'test':
    n_trajectories = config.n_trajectories_test
else:
    raise ValueError("'data_type' argument must be either 'train' or 'test'")

np.random.seed(args.random_seed)

data, n_attempt, n_fail, sol_time, fail_time = generate(
    OCP, config, n_trajectories, controller,
    resolve_failed=True, verbose=args.verbose, suppress_warnings=~args.verbose
)

if args.random_seed is None:
    data['seed'] = np.full_like(data['t'], np.nan)
else:
    data['seed'] = np.full(data['t'].shape, args.random_seed)

n_sol = data['n_trajectories']

print('')
if n_sol >= 1:
    print('\nMean solution time: %1.4f sec' % (sol_time / n_sol))
    print('Total solution time: %1.2f sec' % sol_time)
if n_fail >= 1:
    print('\nMean failure time: %1.4f sec' % (fail_time / n_fail))
    print('Total failure time: %1.2f sec' % fail_time)
    print('\nTotal working time: %1.2f sec' % (sol_time + fail_time))

print('\nTotal data generated: %d' % data['X'].shape[1])

if args.verbose:
    print('')

if not args.verbose or yn_input('Save data?'):
    save_path = os.path.join(data_dir, args.data_type + '.mat')

    try:
        existing_data = scipy.io.loadmat(save_path)

        if not args.verbose or yn_input('Overwrite existing data?'):
            raise FileNotFoundError

        data['n_trajectories'] += existing_data['n_trajectories']
        for key in ['t', 'X', 'dVdX', 'V', 'U', 'seed']:
            if key in data:
                data[key] = np.hstack((existing_data[key], data[key]))

    except FileNotFoundError:
        pass

    scipy.io.savemat(save_path, data)
