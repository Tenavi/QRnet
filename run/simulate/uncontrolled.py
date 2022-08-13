import os
import numpy as np
import scipy.io

from qrnet import simulate

from run.utilities import make_parser

from problem import OCP, config, results_dir

parser = make_parser(rs=True, xd=True)
args = parser.parse_args()

np.random.seed(args.random_seed)

X0 = OCP.sample_X0(1, dist=args.X0_distance)
n_states = OCP.n_states

# ---------------------------------------------------------------------------- #

class NoControl:
    def __init__(self, U_bar):
        self.U_bar = np.reshape(U_bar, (-1,1))

    def eval_U(self, X):
        if X.ndim == 1:
            return np.squeeze(self.U_bar)
        return np.tile(self.U_bar, (1,X.shape[1]))

    def eval_dUdX(self, X):
        zeros = np.zeros((self.U_bar.shape[0], X.shape[0]))
        if X.ndim < 2:
            return zeros

        dUdX = np.expand_dims(zeros, -1)
        dUdX = np.tile(dUdX, (1,1,X.shape[1]))
        return dUdX

controller = NoControl(OCP.U_bar)

# Integrates the open-loop system
t, X, _ = simulate.sim_closed_loop(
    OCP, [0., config.t1_sim], X0, controller,
    solver=config.ode_solver, atol=config.atol, rtol=config.rtol
)

save_dict = {
    'architecture': 'uncontrolled',
    't_NN': t, 'X_NN': X, 'U_NN': controller.eval_U(X)
}

scipy.io.savemat(os.path.join(results_dir, 'sim_data.mat'), save_dict)
