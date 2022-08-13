import os
import numpy as np
import scipy.io

from qrnet.generate import refine

from run.utilities import make_parser, yn_input

from problem import OCP, config, data_dir

parser = make_parser(dt=True, v=True)
args = parser.parse_args()

data = scipy.io.loadmat(os.path.join(data_dir, args.data_type + '.mat'))

refined_data, unrefined_data, sol_time, fail_time = refine(
    OCP, config, data, verbose=args.verbose, suppress_warnings=~args.verbose
)

n_sol = refined_data['n_trajectories']
n_fail = unrefined_data['n_trajectories']

print('')
if n_sol >= 1:
    print('\nMean solution time: %1.4f sec' % (sol_time / n_sol))
    print('Total solution time: %1.2f sec' % sol_time)
if n_fail >= 1:
    print('\nMean failure time: %1.4f sec' % (fail_time / n_fail))
    print('Total failure time: %1.2f sec' % fail_time)
    print('\nTotal working time: %1.2f sec' % (sol_time + fail_time))

print('\nTotal data generated: %d' % data['X'].shape[1])

if yn_input('Save data?'):
    save_path = os.path.join(data_dir, args.data_type + '_refined.mat')
    scipy.io.savemat(save_path, refined_data)

    if n_fail >= 1:
        save_path = os.path.join(data_dir, args.data_type + '_unrefined.mat')
        scipy.io.savemat(save_path, unrefined_data)
        print('Saving failed trajectories to', str(save_path))
