import os
import time
import numpy as np
import scipy.io

from qrnet.controllers import load_NN
from qrnet.validate import eval_errors

from run.utilities import make_parser

from problem import OCP, data_dir, model_dir, results_dir

parser = make_parser(dt=True, mt=True)
args = parser.parse_args()

data_type = args.data_type.lower()
if data_type in ['train', 'test']:
    data = scipy.io.loadmat(os.path.join(data_dir, data_type + '.mat'))
else:
    raise ValueError("'data_type' argument must be either 'train' or 'test'")

controller, _ = load_NN(model_dir, args.timestamp)

start_time = time.time()

pred_data = eval_errors(controller, data, return_predictions=True)

pred_time = time.time() - start_time
n_data = data['X'].shape[0]
print('\nPrediction time for %d data: %.2e sec\n' % (n_data, pred_time))

pred_data = dict(
    (key, val) for key, val in pred_data.items() if key[-4:] == 'pred'
)

pred_data = {**data, **pred_data}

scipy.io.savemat(os.path.join(results_dir, data_type + '_pred.mat'), pred_data)
