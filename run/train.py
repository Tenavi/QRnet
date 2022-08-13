import os
import time
import numpy as np
import scipy.io
from tensorflow import set_random_seed

from qrnet.controllers import create_NN, load_NN
from qrnet.validate import eval_errors, print_errors, find_fixed_point

from run.utilities import make_parser

from problem import OCP, config, data_dir, model_dir

parser = make_parser(a=True, rs=True, mt=True)
args = parser.parse_args()

architecture = getattr(args, 'architecture', 'ControlJacQRnet')

np.random.seed(args.random_seed)
set_random_seed(args.random_seed)

# ---------------------------------------------------------------------------- #

train_data = scipy.io.loadmat(os.path.join(data_dir, 'train.mat'))
test_data = scipy.io.loadmat(os.path.join(data_dir, 'test.mat'))

config.n_trajectories_train = train_data['n_trajectories'][0,0]
config.n_trajectories_test = test_data['n_trajectories'][0,0]

print(
    '\nNumber of training data: %d (%d trajectories)'
    % (train_data['X'].shape[1], config.n_trajectories_train)
)
print(
    'Number of test data: %d (%d trajectories)\n'
    % (test_data['X'].shape[1], config.n_trajectories_test)
)

# ---------------------------------------------------------------------------- #

# Build and train the neural net

if args.timestamp is not None:
    # Load pre-trained model
    controller, _ = load_NN(model_dir, args.timestamp)
else:
    # Initialize the model from scratch
    controller = create_NN(
        architecture,
        OCP.LQR,
        n_hidden=config.n_hidden,
        n_neurons=config.n_neurons,
        activation=config.activation,
        U_star_fun=OCP.make_U_NN
    )

print('\nTraining ' + controller.architecture() + '...')

start_time = time.time()

controller.train(train_data, **config.__dict__)

train_time = time.time() - start_time

print('\nTraining time: %.0f sec\n' % train_time)

# ---------------------------------------------------------------------------- #

# Calculate and print errors

train_errs = eval_errors(controller, train_data)
test_errs = eval_errors(controller, test_data)

print_errors(train_errs, test_errs)

X_star, X_star_deviation, F_star, Jac, eigs, max_eig = find_fixed_point(
    OCP, controller, config.fp_tol
)

# Save results and model parameters
error_dict = {
    'train_time': train_time,
    'fixed_point_deviation': X_star_deviation,
    'max_eig_real': max_eig.real,
    'max_eig_imag': max_eig.imag,
    **dict((key + '_train', val) for key, val in train_errs.items()),
    **dict((key + '_test', val) for key, val in test_errs.items())
}

for config_key in [
        'n_hidden', 'n_neurons',
        'gradient_loss_weight', 'value_loss_weight',
        'n_trajectories_test', 'n_trajectories_train',
        'batch_size', 'n_epochs', 'optimizer', 'optimizer_opts'
    ]:
    error_dict[config_key] = getattr(config, config_key)

controller.save(model_dir, error_dict, args.random_seed)
