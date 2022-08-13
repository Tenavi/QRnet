import os
import numpy as np
import scipy.io

import examples

problem_name = 'van_der_pol'
#problem_name = 'satellite'
#problem_name = 'burgers'
#problem_name = 'uav'

if problem_name == 'van_der_pol':
    from examples.van_der_pol.problem_def import config, MakeOCP
elif problem_name == 'satellite':
    from examples.satellite.problem_def import config, MakeOCP
elif problem_name == 'burgers':
    from examples.burgers.problem_def import config, MakeOCP, N_STATES
    problem_name = os.path.join(problem_name, 'D' + str(N_STATES))
elif problem_name == 'uav':
    from examples.uav.problem_def import config, MakeOCP

OCP = MakeOCP()

problem_dir = os.path.join('examples', problem_name)
data_dir = os.path.join(problem_dir, 'data')
model_dir = os.path.join(problem_dir, 'models')
results_dir = os.path.join(problem_dir, 'results')

for dir_name in [data_dir, model_dir, results_dir]:
    os.makedirs(dir_name, exist_ok=True)

# Save model parameters for use in MATLAB
params_dict = OCP.get_params()
params_path = os.path.join(problem_dir, 'params.mat')
try:
    # Check to make sure this isn't a duplicate (annoying for git)
    old_params = scipy.io.loadmat(params_path)

    for key, param in params_dict.items():
        old_param = np.reshape(old_params.get(key), np.shape(param))
        if not np.all(np.isclose(param, old_param)):
            if not np.isnan(param) and not np.isnan(old_param):
                print(key, param, old_param)
                raise
except:
    scipy.io.savemat(params_path, params_dict)
