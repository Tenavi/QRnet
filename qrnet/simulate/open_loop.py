import numpy as np
from copy import deepcopy
import warnings
from scipy.integrate import solve_bvp

import pylgr

class OpenLoopSolver:
    def __init__(self, OCP, **kwargs):
        self.OCP = OCP
        self.sol = {}

    def solve(self, t=None, X=None, U=None, dVdX=None, V=None, verbose=0):
        raise NotImplementedError

    def continuous_sol(self, t):
        raise NotImplementedError

    def check_converged(self, tol):
        raise NotImplementedError

    def extend_horizon(self):
        raise NotImplementedError

class IndirectSolver(OpenLoopSolver):
    def __init__(
            self, OCP, tol=1e-05, t1_scale=3/2, t1_max=np.inf, max_nodes=1000
        ):
        self.tol = tol
        self.t1_scale = t1_scale
        self.t1_max = t1_max
        self.max_nodes = max_nodes

        self.bvp_sol = None
        self.bc = None

        super().__init__(OCP)

    def solve(self, t=None, X=None, U=None, dVdX=None, V=None, verbose=0):
        X = X.reshape(self.OCP.n_states, -1)
        dVdX = dVdX.reshape(self.OCP.n_states, -1)
        V = V.reshape(1, -1)

        X_aug = np.vstack((X, dVdX, V))

        if self.bc is None:
            self.bc = self.OCP.make_bc(X[:,0])

        self.bvp_sol = solve_bvp(
            self.OCP.bvp_dynamics, self.bc, t, X_aug,
            tol=self.tol, max_nodes=self.max_nodes, verbose=verbose
        )

        self.sol['t'] = self.bvp_sol.x
        self.sol['X'] = self.bvp_sol.y[:self.OCP.n_states]
        self.sol['dVdX'] = self.bvp_sol.y[self.OCP.n_states:-1]
        self.sol['V'] = self.bvp_sol.y[-1:]
        self.sol['U'] = self.OCP.U_star(self.sol['X'], self.sol['dVdX'])

    def continuous_sol(self, t):
        X_aug = self.bvp_sol.sol(t)

        X = X_aug[:self.OCP.n_states]
        dVdX = X_aug[self.OCP.n_states:-1]
        V = X_aug[-1:]

        U = self.OCP.U_star(X, dVdX)

        return {'X': X, 'U': U, 'dVdX': dVdX, 'V': V}

    def check_converged(self, tol):
        '''
        Check if the running cost L and vector field F at final time are smaller
        than a given tolerance, to see if the BVP is converged.

        Parameters
        ----------

        Returns
        -------
        '''
        if self.bvp_sol is None or not self.bvp_sol.success:
            return False

        L = self.OCP.running_cost(self.sol['X'][:,-1], self.sol['U'][:,-1])
        F = self.OCP.bvp_dynamics(self.sol['t'][-1:], self.bvp_sol.y[:,-1:])
        F = np.linalg.norm(F[:self.OCP.n_states])

        return L <= tol and F <= tol

    def extend_horizon(self):
        if self.bvp_sol is None:
            return False
        # Cannot extend horizon if exceeded number of mesh nodes or maximum time
        if self.bvp_sol.status == 1 or self.sol['t'][-1] >= self.t1_max:
            return False

        self.sol['t'][-1] = np.minimum(
            self.t1_max, self.sol['t'][-1]*self.t1_scale
        )

        return True

class DirectSolver(OpenLoopSolver):
    def __init__(
            self, OCP, tol=1e-06, tol_scale=0.1,
            n_init_nodes=16, n_add_nodes=16, max_nodes=64, max_iter=500
        ):
        self.tol = tol
        self.tol_scale = tol_scale
        self.n_nodes = n_init_nodes
        self.n_add_nodes = n_add_nodes
        self.max_nodes = max_nodes
        self.max_iter = max_iter

        self.ps_sol = None

        super().__init__(OCP)

    def solve(self, t=None, X=None, U=None, dVdX=None, V=None, verbose=0):
        '''
        Wrapper of pylgr.solve_ocp

        Parameters
        ----------
        verbose : {0, 1, 2}, default=0
            Level of algorithm's verbosity:
                * 0 (default) : work silently.
                * 1 : display a termination report.
                * 2 : display progress during iterations.
        '''
        self.ps_sol = pylgr.solve_ocp(
            self.OCP.dynamics, self.OCP.running_cost, t, X, U,
            U_lb=self.OCP.U_lb, U_ub=self.OCP.U_ub,
            dynamics_jac=self.OCP.jacobians,
            cost_grad=self.OCP.running_cost_gradient,
            tol=self.tol, n_nodes=self.n_nodes, maxiter=self.max_iter,
            verbose=verbose
        )

        self.sol['t'] = self.ps_sol.t.flatten()
        self.sol['X'] = self.ps_sol.X
        self.sol['dVdX'] = self.ps_sol.dVdX
        self.sol['V'] = self.ps_sol.V
        self.sol['U'] = self.ps_sol.U

    def continuous_sol(self, t):
        return {
            'X': np.atleast_2d(self.ps_sol.sol_X(t)),
            'dVdX': np.atleast_2d(self.ps_sol.sol_dVdX(t)),
            'V': np.atleast_2d(self.ps_sol.sol_V(t)),
            'U': np.atleast_2d(self.ps_sol.sol_U(t))
        }

    def check_converged(self, tol):
        '''
        Check if the running cost L and vector field F at final time are smaller
        than a given tolerance, to see if the OCP is converged.

        Parameters
        ----------

        Returns
        -------
        '''
        if self.ps_sol is None or not self.ps_sol.success:
            return False

        if np.any(self.ps_sol.residuals > tol):
            return False

        L = self.OCP.running_cost(self.sol['X'][:,-1], self.sol['U'][:,-1])
        F = self.OCP.dynamics(self.sol['X'][:,-1], self.sol['U'][:,-1])
        F = np.linalg.norm(F)

        return L <= tol and F <= tol

    def extend_horizon(self):
        if self.ps_sol is None:
            return False
        # Cannot extend horizon if exceeded number of mesh nodes
        if self.n_nodes >= self.max_nodes:
            return False

        self.n_nodes = min(self.max_nodes, self.n_nodes+self.n_add_nodes)
        self.tol = self.tol * self.tol_scale

        return True

# ---------------------------------------------------------------------------- #

def solve_ocp(
        OCP, config,
        t_guess=None, X_guess=None, U_guess=None, dVdX_guess=None, V_guess=None,
        solve_to_converge=False, verbose=0, suppress_warnings=True
    ):
    if config.ocp_solver == 'indirect':
        solver = IndirectSolver(
            OCP,
            tol=config.indirect_tol,
            t1_scale=config.t1_scale,
            t1_max=config.t1_max,
            max_nodes=config.indirect_max_nodes
        )
    elif config.ocp_solver == 'direct':
        solver = DirectSolver(
            OCP,
            tol=config.direct_tol,
            tol_scale=config.direct_tol_scale,
            n_init_nodes=config.direct_n_init_nodes,
            n_add_nodes=config.direct_n_add_nodes,
            max_nodes=config.direct_max_nodes,
            max_iter=config.direct_max_slsqp_iter
        )
    elif config.ocp_solver == 'direct_to_indirect':
        config_copy = deepcopy(config)
        config_copy.ocp_solver = 'direct'

        direct_start, _, _ = solve_ocp(
            OCP,
            config_copy,
            t_guess=t_guess,
            X_guess=X_guess,
            U_guess=U_guess,
            dVdX_guess=dVdX_guess,
            V_guess=V_guess,
            solve_to_converge=True,
            verbose=verbose,
            suppress_warnings=suppress_warnings
        )

        config_copy.ocp_solver = 'indirect'

        idx = direct_start['t'] <= t_guess[-1]
        dVdX_guess = direct_start['dVdX'][:,idx]
        dVdX_guess[:,-1] = 0.

        return solve_ocp(
            OCP,
            config_copy,
            t_guess=direct_start['t'][idx],
            X_guess=direct_start['X'][:,idx],
            U_guess=direct_start['U'][:,idx],
            dVdX_guess=dVdX_guess,
            V_guess=direct_start['V'][:,idx],
            solve_to_converge=solve_to_converge,
            verbose=verbose,
            suppress_warnings=suppress_warnings
        )
    else:
        raise ValueError(
            'config.ocp_solver must be one of "direct", "indirect", or "direct_to_indirect"'
        )

    with warnings.catch_warnings():
        if suppress_warnings:
            np.seterr(over='warn', divide='warn', invalid='warn')
            warnings.filterwarnings("ignore", category=RuntimeWarning)

        solver.solve(
            t=t_guess, X=X_guess, U=U_guess, dVdX=dVdX_guess, V=V_guess,
            verbose=verbose
        )
        converged = solver.check_converged(config.fp_tol)

        # Solves the OCP over an extended time interval until convergece
        # conditions are satisfied
        if solve_to_converge and not converged:
            if suppress_warnings:
                # If we don't want to see warnings, treat these as errors so the
                # try context will catch them and mark the trajectory as not
                # converged instead of printing out the warning.
                warnings.filterwarnings("error", category=RuntimeWarning)

            try:
                while not converged and solver.extend_horizon():
                    solver.solve(**solver.sol, verbose=verbose)
                    converged = solver.check_converged(config.fp_tol)
            except RuntimeWarning:
                pass

        return solver.sol, solver.continuous_sol, converged

# ---------------------------------------------------------------------------- #

def clip_trajectory(t, X, U, criteria):
    '''
    Go backwards in time and check to see when a function (typically the
    running cost or vector field norm) is sufficiently small.

    Parameters
    ----------
    criteria : callable

    Returns
    -------
    converged_idx : int
        Integer such that X[:,converged_idx], U[:,converged_idx] is the first
        state-control pair for which criteria(X, U) < tol.
    converged : bool
        True if some pair X, U satisfied criteria(X, U) < tol, False if no
        such pair was found.
    '''
    converged_idx = criteria(X, U).flatten()

    converged = converged_idx.any()

    if converged:
        converged_idx = np.min(np.argwhere(converged_idx))
    else:
        converged_idx = t.shape[0] - 1

    return converged_idx, converged
