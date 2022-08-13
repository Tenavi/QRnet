import numpy as np
from tqdm import tqdm

from .closed_loop import sim_to_converge
from .open_loop import solve_ocp, clip_trajectory

def monte_carlo(
        OCP, config, controller,
        X0_distance=None, random_seed=None, X0_pool=None,
        solve_open_loop=False, verbose=0, suppress_warnings=True
    ):
    '''
    Parameters
    ----------
    OCP : instance of QRnet.problem_template.TemplateOCP
    config : instance of QRnet.problem_template.MakeConfig
    controller : instantiated BaseNN subclass
    X0_distance : float, optional
        Norm to use for random initial conditions
    random_seed : int, optional
        Seed for numpy random number generator
    solve_open_loop : bool, default=False
        Set to True to solve the open loop OCP for each initial condition.
    suppress_warnings : bool, default=True
        If True, treat numpy warnings as OCP failures.

    Returns
    -------
    results_dict : dict containing
        'seed' : random_seed
        'X0_pool' : (n_states, n_MC) array
            Initial conditions used for each Monte Carlo simulation.
        'init_dists' : (n_MC,) array
            Vector of initial condition norms
        'final_dists': (n_MC,) array
            Vector of norms of NN-controlled trajectories at final time
        'NN_final_times' : (n_MC,) array
            Vector of times for NN-controlled trajectories to reach equilibrium
        'NN_costs' : (n_MC,) array
            Vector of accumulated costs by NN-controlled trajectories
        'opt_final_times' : (n_MC,) array
            Vector of times for optimal trajectories to reach equilibrium
        'opt_costs' : (n_MC,) array
            Vector of optimal costs
        'ocp_converged' : (n_MC,) bool array
            Whether or not each OCP was solved correctly
    '''

    n_MC = config.n_trajectories_MC

    if X0_pool is None:
        np.random.seed(random_seed)
        X0_pool = OCP.sample_X0(n_MC, dist=X0_distance)

    ocp_converged = np.zeros(n_MC, dtype=bool)

    init_dists = np.empty(n_MC)
    final_dists = np.empty(n_MC)

    NN_final_times = np.full_like(init_dists, np.inf)
    NN_costs = np.full_like(init_dists, np.inf)

    # Default final time value is -1
    # Not possible under normal circumstances, helps with catching errors later
    opt_final_times = np.full_like(NN_final_times, -1.)
    opt_costs = np.full_like(NN_costs, -1.)

    def closed_converged(X, U):
        return np.linalg.norm(OCP.dynamics(X, U), axis=0) < config.fp_tol
    def open_converged(X, U):
        return OCP.running_cost(X, U) < config.fp_tol

    events = OCP.make_integration_events()

    # ------------------------------------------------------------------------ #

    for i in tqdm(range(n_MC)):
        X0 = X0_pool[:,i]

        init_dists[i] = OCP.norm(X0)

        # Integrates the closed-loop system
        t, X, ode_converged = sim_to_converge(
            OCP.dynamics, OCP.closed_loop_jacobian, controller, X0, config,
            events=events
        )
        V, dVdX, U = controller.bvp_guess(X)

        k, _ = clip_trajectory(t, X, U, closed_converged)

        final_dists[i] = OCP.norm(X[:,k])
        # If converged fill in results, otherwise leave as infinity
        if ode_converged:
            NN_final_times[i] = t[k]
            NN_costs[i] = OCP.compute_cost(t, X, U).flatten()[-1]

        # -------------------------------------------------------------------- #

        if solve_open_loop:
            ocp_sol, cont_ocp_sol, ocp_converged[i] = solve_ocp(
                OCP, config,
                t_guess=t, X_guess=X, U_guess=U, dVdX_guess=dVdX, V_guess=V,
                solve_to_converge=True,
                verbose=verbose, suppress_warnings=suppress_warnings
            )

            t = np.concatenate((ocp_sol['t'].flatten(), t.flatten()))
            t = np.unique(t)
            ocp_sol = cont_ocp_sol(t)

            k, _ = clip_trajectory(
                t, ocp_sol['X'], ocp_sol['U'], open_converged
            )

            opt_costs[i] = ocp_sol['V'].flatten()[0]
            opt_final_times[i] = t[k]

    # ------------------------------------------------------------------------ #

    results_dict = {
        'seed': random_seed,
        'X0_pool': X0_pool,
        'init_dists': init_dists,
        'final_dists': final_dists,
        'NN_final_times': NN_final_times,
        'NN_costs': NN_costs,
        'opt_final_times': opt_final_times,
        'opt_costs': opt_costs,
        'ocp_converged': ocp_converged
    }
    if random_seed is None:
        results_dict['seed'] = -1

    if solve_open_loop:
        print('\n%d/%d OCPs converged.\n' % (ocp_converged.sum(), n_MC))

    return results_dict
