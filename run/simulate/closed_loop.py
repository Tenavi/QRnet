import os
import time
import numpy as np
import scipy.io
from scipy.interpolate import interp1d

from qrnet import simulate
from qrnet.controllers import load_NN

from run.utilities import make_parser

from problem import OCP, config, model_dir, results_dir, data_dir

parser = make_parser(mt=True, rs=True, xd=True, io=True)
args = parser.parse_args()

controller, timestamp = load_NN(model_dir, args.timestamp)

if args.import_open_loop:
    test_data = scipy.io.loadmat(os.path.join(data_dir, 'test.mat'))

    if args.X0_distance is not None:
        raise ValueError('Cannot specify X0_distance if importing test data.')

    t0_idx = np.where(test_data['t'] == 0.)[1]
    t0_idx = np.append(t0_idx, test_data['t'].shape[1])

    if args.random_seed is None or args.random_seed > t0_idx.shape[0]:
        np.random.seed(args.random_seed)
        k0 = np.random.choice(t0_idx.shape[0]-1, 1)
        print('Using test trajectory #%d' % k0)
    else:
        k0 = args.random_seed

    k1 = int(t0_idx[k0+1])
    k0 = int(t0_idx[k0])
    X0 = test_data['X'][:,k0]
else:
    np.random.seed(args.random_seed)
    X0 = OCP.sample_X0(1, dist=args.X0_distance)

n_states = OCP.n_states

# ---------------------------------------------------------------------------- #

# Integrates the closed-loop system (LQR controller and NN controllers)
print('Integrating closed loop system (LQR)...')
start_time = time.time()

t_LQR, X_LQR, LQR_converged = simulate.sim_to_converge(
    OCP.dynamics, OCP.closed_loop_jacobian, OCP.LQR, X0, config,
    events=OCP.make_integration_events()
)

print('Integration time (LQR) %.4f s' % (time.time() - start_time))

print('Integrating closed loop system (%s)...' % controller.architecture())

start_time = time.time()

t_NN, X_NN, NN_converged = simulate.sim_to_converge(
    OCP.dynamics, OCP.closed_loop_jacobian, controller, X0, config,
    events=OCP.make_integration_events()
)

print(
    'Integration time (%s) %.4f s'
    % (controller.architecture(), time.time() - start_time)
)

# Compute costates, controls, and cost
V_LQR, dVdX_LQR, U_LQR = OCP.LQR.bvp_guess(X_LQR)
J_LQR = OCP.compute_cost(t_LQR, X_LQR, U_LQR)

V_NN, dVdX_NN, U_NN = controller.bvp_guess(X_NN)
J_NN = OCP.compute_cost(t_NN, X_NN, U_NN)

save_dict = {
    'architecture': controller.architecture(),
    'timestamp': int(timestamp),
    'LQR_converged': LQR_converged,
    'NN_converged': NN_converged,
    't_LQR': t_LQR, 'X_LQR': X_LQR, 'U_LQR': U_LQR, 'J_LQR': J_LQR,
    'V_LQR': V_LQR, 'dVdX_LQR': dVdX_LQR,
    't_NN': t_NN, 'X_NN': X_NN, 'U_NN': U_NN, 'J_NN': J_NN,
    'V_NN': V_NN, 'dVdX_NN': dVdX_NN
}

# ---------------------------------------------------------------------------- #

# Solves the two-point BVP with LQR and NN initial guesses
def _linear_guess(t, Y0, Y1):
    Y = np.hstack((Y0.reshape(-1,1), Y1.reshape(-1,1)))
    Y = interp1d([0., config.t1_sim], Y)
    return Y(t)

def _solve_ocp_from_guess(t, X, U, dVdX, V):
    if OCP.running_cost(X[:,-1], U[:,-1]) > config.fp_tol:
        print('Initial guess failed to converge. Using linear interpolation.')
        t = np.linspace(0., config.t1_sim)
        X = _linear_guess(t, X0, OCP.X_bar)
        U = _linear_guess(t, U[:,:1], OCP.U_bar)
        dVdX = np.zeros_like(X)
        V = _linear_guess(t, V[:,:1], np.zeros((1,1)))

    return simulate.solve_ocp(
        OCP, config,
        t_guess=t, X_guess=X, U_guess=U, dVdX_guess=dVdX, V_guess=V,
        solve_to_converge=True, verbose=2-(config.ocp_solver=='direct')
    )

if args.import_open_loop:
    start_time = time.time()

    ocp_sol, _, _ = _solve_ocp_from_guess(
        t=test_data['t'][0,k0:k1],
        X=test_data['X'][:,k0:k1],
        U=test_data['U'][:,k0:k1],
        dVdX=test_data['dVdX'][:,k0:k1],
        V=test_data['V'][0,k0:k1]
    )

    print('OCP solution time: %.2f s' % (time.time() - start_time))
    J_opt = ocp_sol['V'].flatten()[::-1]
else:
    start_time = time.time()

    _, ocp_sol_LQR, LQR_ocp_converged = _solve_ocp_from_guess(
        t_LQR, X_LQR, U_LQR, dVdX_LQR, V_LQR
    )
    ocp_sol_LQR = ocp_sol_LQR(t_LQR)
    ocp_sol_LQR['t'] = t_LQR

    print('OCP solution time: %.2f s' % (time.time() - start_time))
    start_time = time.time()

    _, ocp_sol_NN, NN_ocp_converged = _solve_ocp_from_guess(
        t_NN, X_NN, U_NN, dVdX_NN, V_NN
    )
    ocp_sol_NN = ocp_sol_NN(t_NN)
    ocp_sol_NN['t'] = t_NN

    print('OCP solution time: %.2f s' % (time.time() - start_time))

    J_opt_LQR = ocp_sol_LQR['V'].flatten()[::-1]
    J_opt_NN = ocp_sol_NN['V'].flatten()[::-1]

    # Uses the better BVP solution in case of multiple local minima
    if LQR_ocp_converged and NN_ocp_converged:
        if J_opt_LQR[-1] < J_opt_NN[-1]:
            J_opt, ocp_sol = J_opt_LQR, ocp_sol_LQR
        else:
            J_opt, ocp_sol = J_opt_NN, ocp_sol_NN
    elif LQR_ocp_converged:
        J_opt, ocp_sol = J_opt_LQR, ocp_sol_LQR
    elif NN_ocp_converged:
        J_opt, ocp_sol = J_opt_NN, ocp_sol_NN
    elif J_opt_LQR[-1] < J_opt_NN[-1]:
        J_opt, ocp_sol = J_opt_LQR, ocp_sol_LQR
    else:
        J_opt, ocp_sol = J_opt_NN, ocp_sol_NN

for key, val in ocp_sol.items():
    save_dict[key + '_opt'] = val
save_dict['H_opt'] = OCP.Hamiltonian(
    ocp_sol['X'], ocp_sol['U'], ocp_sol['dVdX']
)
save_dict['J_opt'] = J_opt

# -----------------------------------------------------------------------------#

if LQR_converged:
    print('LQR cost: %.2f' % J_LQR[-1])
else:
    print('LQR cost: infinite (%.2f)' % J_LQR[-1])
if NN_converged:
    print('NN cost: %.2f' % J_NN[-1])
else:
    print('NN cost: infinite (%.2f)' % J_NN[-1])

if J_opt[-1] < np.infty:
    print('Optimal cost: %.2f \n' % J_opt[-1])
    if LQR_converged:
        print('LQR sub-optimality: {J:.2f} %'.format(
            J=np.maximum(0., 100.*(J_LQR[-1]/J_opt[-1] - 1.))
        ))
    else:
        print('LQR sub-optimality: infinite ({J:.2f}) %'.format(
            J=np.maximum(0., 100.*(J_LQR[-1]/J_opt[-1] - 1.))
        ))
    if NN_converged:
        print('NN sub-optimality: {J:.2f} %'.format(
            J=np.maximum(0., 100.*(J_NN[-1]/J_opt[-1] - 1.))
        ))
    else:
        print('NN sub-optimality: infinite ({J:.2f}) %'.format(
            J=np.maximum(0., 100.*(J_NN[-1]/J_opt[-1] - 1.))
        ))

scipy.io.savemat(os.path.join(results_dir, 'sim_data.mat'), save_dict)
