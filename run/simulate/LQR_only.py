import os
import time
import numpy as np
import scipy.io
from scipy.interpolate import interp1d

from qrnet import simulate

from run.utilities import make_parser

from problem import OCP, config, results_dir

parser = make_parser(rs=True, xd=True)
args = parser.parse_args()

np.random.seed(args.random_seed)

X0 = OCP.sample_X0(1, dist=args.X0_distance)
n_states = OCP.n_states

# ---------------------------------------------------------------------------- #

print('Integrating closed loop system...')
start_time = time.time()

# Integrates the closed-loop system (LQR controller)
t_LQR, X_LQR, _ = simulate.sim_closed_loop(
    OCP.dynamics,
    OCP.closed_loop_jacobian,
    OCP.LQR,
    [0., config.t1_sim],
    X0,
    solver=config.ode_solver,
    atol=config.atol,
    rtol=config.rtol
)

print('Integration time %.4f s' % (time.time() - start_time))

# Compute costates, controls, and cost
V_LQR, dVdX_LQR, U_LQR = OCP.LQR.bvp_guess(X_LQR)
J_LQR = OCP.compute_cost(t_LQR, X_LQR, U_LQR)

save_dict = {
    't_LQR': t_LQR, 'X_LQR': X_LQR, 'U_LQR': U_LQR,
    'V_LQR': V_LQR, 'dVdX_LQR': dVdX_LQR, 'J_LQR': J_LQR
}

# ---------------------------------------------------------------------------- #

def _linear_guess(t, Y0, Y1):
    Y = np.hstack((Y0.reshape(-1,1), Y1.reshape(-1,1)))
    Y = interp1d([0., config.t1_sim], Y)
    return Y(t)

if OCP.running_cost(X_LQR[:,-1], U_LQR[:,-1]) > config.fp_tol:
    print('Initial guess failed to converge. Trying linear interpolation.')
    t_guess = np.linspace(0., config.t1_sim)
    X_guess = _linear_guess(t_guess, X0, OCP.X_bar)
    U_guess = _linear_guess(t_guess, U_LQR[:,:1], OCP.U_bar)
    dVdX_guess = _linear_guess(t_guess, dVdX_LQR[:,:1], np.zeros_like(OCP.X_bar))
    V_guess = _linear_guess(t_guess, V_LQR[:,:1], np.zeros((1,1)))
else:
    t_guess = t_LQR
    X_guess = X_LQR
    U_guess = U_LQR
    dVdX_guess = dVdX_LQR
    V_guess = V_LQR

start_time = time.time()

_, ocp_sol, ocp_converged = simulate.solve_ocp(
    OCP, config,
    t_guess=t_guess, X_guess=X_guess, U_guess=U_guess,
    dVdX_guess=dVdX_guess, V_guess=V_guess,
    solve_to_converge=True, verbose=2
)

print('OCP solution time: %.2f s' % (time.time() - start_time))

ocp_sol = ocp_sol(t_LQR)
ocp_sol['t'] = t_LQR
for key, val in ocp_sol.items():
    save_dict[key + '_opt'] = val
save_dict['H_opt'] = OCP.Hamiltonian(
    ocp_sol['X'], ocp_sol['U'], ocp_sol['dVdX']
)
save_dict['J_opt'] = ocp_sol['V'].flatten()[::-1]

if ocp_converged:
    print('OCP successfully converged')
else:
    print('OCP failed to converge')

print('LQR cost: %.2f' % J_LQR[-1])
print('Optimal cost: %.2f' % save_dict['J_opt'][-1])
print('LQR sub-optimality: {J:.2f} %'.format(
    J=np.maximum(0., 100.*(J_LQR[-1]/save_dict['J_opt'][-1] - 1.))
))

scipy.io.savemat(os.path.join(results_dir, 'sim_data.mat'), save_dict)
