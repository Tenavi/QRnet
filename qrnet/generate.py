import time
import warnings
import numpy as np
from scipy.interpolate import interp1d

from qrnet import simulate

_headers = (
    '\n attempted |  solved   |  desired  ',
    '-----------------------------------'
)

def generate(
        OCP, config, n_trajectories, controller=None, resolve_failed=True,
        verbose=0, suppress_warnings=True
    ):
    '''
    Generate data for an OCP by solving n_trajectories open loop OCPs. Uses LQR
    or a provided NN controller to warm start the BVP. Returns the portion of
    each trajectory up to the time taken for the running cost to approximately
    reach zero.

    Parameters
    ----------
    OCP : object
        Instance of QRnet.problem_template.TemplateOCP
    config : object
        Instance of QRnet.problem_template.MakeConfig
    n_trajectories: int
        Number of optimal trajectories to generate
    controller : object, optional
        Instantiated BaseNN subclass. If None (default), use LQR for warm start
    resolve_failed : bool, default=True
        If True, continue attempting to solve BVPs until get n_trajectories
        successful solutions
    verbose : int, default=0
        See scipy.integrate.solve_bvp
    suppress_warnings : bool, default=True
        If True, treat numpy warnings as BVP failures

    Returns
    -------
    data : dict
        Open loop optimal control data containing
        n_trajectories : int
            Number of successfully integrated BVPs
        t : (1, n_data) array
            Time instances of each data point
        X : (n_states, n_data) array
            Optimal states of each data point
        dVdX : (n_states, n_data) array
            Costates i.e. value gradient at each data point, if available
        V : (1, n_data) array
            Optimal cost at each state
        U : (n_controls, n_data) array
            Optimal control for each state
    n_attempt : int
        Number of attempted BVPs
    n_fail : int
        Number of failed solution attempts
    sol_time : float
        Total time of successful solution attempts in seconds
    fail_time : float
        Total time of failed solution attempts in seconds
    '''
    data = {}

    events = OCP.make_integration_events()

    def open_converged(X, U):
        return OCP.running_cost(X, U) < config.fp_tol

    X0_pool = OCP.sample_X0(n_trajectories).reshape(OCP.n_states, -1)

    n_attempt = 0
    n_sol = 0
    n_fail = 0
    sol_time = []
    fail_time = []

    if resolve_failed:
        n_track = lambda : n_sol
    else:
        n_track = lambda : n_attempt

    # ------------------------------------------------------------------------ #

    with warnings.catch_warnings():
        if suppress_warnings:
            np.seterr(over='warn', divide='warn', invalid='warn')
            warnings.filterwarnings('error', category=RuntimeWarning)
        warnings.filterwarnings('error', category=UserWarning)

        print('\nSolving open loop OCPs...')
        for header in _headers:
            print(header)
        w = str(len('attempted') + 2)
        row = '{att:^' + w + 'd}|{sol:^' + w + 'd}|{des:^' + w + 'd}'

        while n_track() < n_trajectories:
            X0 = X0_pool[:, n_track()]

            n_attempt += 1

            start_time = time.time()

            try:
                # Integrates the closed-loop system to warm start the OCP solver
                t, X, ode_converged = simulate.sim_to_converge(
                    OCP.dynamics, OCP.closed_loop_jacobian, controller, X0,
                    config, events=events
                )

                if ode_converged:
                    V, dVdX, U = controller.bvp_guess(X)
                else:
                    # Use linear interpolation if the warm start controller failed
                    # to stabilize the system
                    t = np.linspace(
                        0., config.t1_sim, config.direct_n_init_nodes
                    )
                    X = np.hstack((X0.reshape(-1,1), OCP.X_bar))
                    X = interp1d([0., config.t1_sim], X)(t)

                    V = np.zeros_like(t)
                    dVdX = np.zeros_like(X)
                    U = np.tile(OCP.U_bar, (1,X.shape[1]))

                # Solves the two-point BVP until to convergence to infinite
                # horizon approximation
                ocp_sol, cont_ocp_sol, ocp_converged = simulate.solve_ocp(
                    OCP, config,
                    t_guess=t, X_guess=X, U_guess=U, dVdX_guess=dVdX, V_guess=V,
                    solve_to_converge=True,
                    verbose=verbose, suppress_warnings=suppress_warnings
                )

                if ocp_converged:
                    sol_time.append(time.time() - start_time)

                    n_sol += 1

                    t = np.concatenate((ocp_sol['t'].flatten(), t.flatten()))
                    ocp_sol['t'] = np.unique(t)
                    ocp_sol.update(cont_ocp_sol(ocp_sol['t']))

                    # Clips the trajectory to when the running cost first gets
                    # close to zero, reducing the concentration of data near the
                    # equilibrium
                    keep_idx, _ = simulate.open_loop.clip_trajectory(
                        ocp_sol['t'], ocp_sol['X'], ocp_sol['U'], open_converged
                    )

                    for key, new_data in ocp_sol.items():
                        if key not in data:
                            data[key] = []
                        data[key].append(np.atleast_2d(new_data)[:,:keep_idx+1])
                else:
                    warnings.warn(UserWarning())

            except (UserWarning, RuntimeWarning):
                fail_time.append(time.time() - start_time)

                n_fail += 1

                # Resample the failed initial condition
                if resolve_failed:
                    X0_pool[:,n_sol] = OCP.sample_X0(1)

            if verbose:
                for header in _headers:
                    print(header)
            print(
                row.format(att=n_attempt, sol=n_sol, des=n_trajectories),
                end='\r'
            )

    for key, val in data.items():
        data[key] = np.hstack(val)
    data['n_trajectories'] = n_sol

    sol_time, fail_time = np.sum(sol_time), np.sum(fail_time)

    return data, n_attempt, n_fail, sol_time, fail_time

def refine(OCP, config, data, verbose=0, suppress_warnings=True):
    '''
    Given an existing open loop data set, resolve the open loop OCP using the
    previously generated data as initial guesses. Used when refining solutions
    with an indirect method or higher tolerances.

    Parameters
    ----------
    OCP : object
        Instance of QRnet.problem_template.TemplateOCP
    config : object
        Instance of QRnet.problem_template.MakeConfig
    data : dict
        Open loop optimal control data containing
        n_trajectories : int
            Number of successfully integrated BVPs
        t : (1, n_data) array
            Time instances of each data point
        X : (n_states, n_data) array
            Optimal states of each data point
        dVdX : (n_states, n_data) array
            Costates i.e. value gradient at each data point, if available
        V : (1, n_data) array
            Optimal cost at each state
        U : (n_controls, n_data) array
            Optimal control for each state
    verbose : int, default=0
        See scipy.integrate.solve_bvp
    suppress_warnings : bool, default=True
        If True, treat numpy warnings as BVP failures

    Returns
    -------
    data : dict
        Open loop optimal control data containing
        n_trajectories : int
            Number of successfully integrated BVPs
        t : (1, n_data) array
            Time instances of each data point
        X : (n_states, n_data) array
            Optimal states of each data point
        dVdX : (n_states, n_data) array
            Costates i.e. value gradient at each data point, if available
        V : (1, n_data) array
            Optimal cost at each state
        U : (n_controls, n_data) array
            Optimal control for each state
    sol_time : float
        Total time of successful solution attempts in seconds
    fail_time : float
        Total time of failed solution attempts in seconds
    '''
    n_trajectories = int(data['n_trajectories'])

    refined_data = {}
    unrefined_data = {}

    t0_idx = np.where(data['t'] == 0.)[1]
    t0_idx = np.append(t0_idx, data['t'].shape[1])

    def open_converged(X, U):
        return OCP.running_cost(X, U) < config.fp_tol

    n_attempt = 0
    n_sol = 0
    n_fail = 0
    sol_time = []
    fail_time = []

    # ------------------------------------------------------------------------ #

    with warnings.catch_warnings():
        if suppress_warnings:
            np.seterr(over='warn', divide='warn', invalid='warn')
            warnings.filterwarnings('error', category=RuntimeWarning)
        warnings.filterwarnings('error', category=UserWarning)

        print('\nSolving open loop OCPs...')
        for header in _headers:
            print(header)
        w = str(len('attempted') + 2)
        row = '{att:^' + w + 'd}|{sol:^' + w + 'd}|{des:^' + w + 'd}'

        for i in range(n_trajectories):
            k0 = t0_idx[i]
            k1 = t0_idx[i+1]

            t = data['t'][0,k0:k1]
            X = data['X'][:,k0:k1]
            dVdX = data['dVdX'][:,k0:k1]
            V = data['V'][0,k0:k1]
            U = data['U'][:,k0:k1]

            seed = data['seed'][0,k0:k1]

            n_attempt += 1

            start_time = time.time()

            try:
                # Solves the two-point BVP until to convergence to infinite
                # horizon approximation
                ocp_sol, cont_ocp_sol, ocp_converged = simulate.solve_ocp(
                    OCP, config,
                    t_guess=t, X_guess=X, U_guess=U, dVdX_guess=dVdX, V_guess=V,
                    solve_to_converge=True,
                    verbose=verbose, suppress_warnings=suppress_warnings
                )

                if ocp_converged:
                    sol_time.append(time.time() - start_time)

                    n_sol += 1

                    t = np.concatenate((ocp_sol['t'].flatten(), t))
                    ocp_sol['t'] = np.unique(t)
                    ocp_sol.update(cont_ocp_sol(ocp_sol['t']))
                    ocp_sol['seed'] = np.full(ocp_sol['t'].shape, seed[0])

                    # Clips the trajectory to when the running cost first gets
                    # close to zero, reducing the concentration of data near the
                    # equilibrium
                    keep_idx, _ = simulate.open_loop.clip_trajectory(
                        ocp_sol['t'], ocp_sol['X'], ocp_sol['U'], open_converged
                    )

                    for key, new_data in ocp_sol.items():
                        if key not in refined_data:
                            refined_data[key] = []
                        refined_data[key].append(
                            np.atleast_2d(new_data)[:,:keep_idx+1]
                        )
                else:
                    warnings.warn(UserWarning())

            except (UserWarning, RuntimeWarning):
                fail_time.append(time.time() - start_time)

                n_fail += 1

                for key, old_data in zip(
                        ['t','X','dVdX','V','U','seed'],
                        [t, X, dVdX, V, U, seed]
                    ):
                    if key not in unrefined_data:
                        unrefined_data[key] = []
                    unrefined_data[key].append(np.atleast_2d(old_data))

            if verbose:
                for header in _headers:
                    print(header)
            print(
                row.format(att=n_attempt, sol=n_sol, des=n_trajectories),
                end='\r' if not verbose else None
            )

    for key in ['t','X','dVdX','V','U','seed']:
        if key in refined_data:
            refined_data[key] = np.hstack(refined_data[key])
        if key in unrefined_data:
            unrefined_data[key] = np.hstack(unrefined_data[key])

    refined_data['n_trajectories'] = n_sol
    unrefined_data['n_trajectories'] = n_fail

    sol_time, fail_time = np.sum(sol_time), np.sum(fail_time)

    return refined_data, unrefined_data, sol_time, fail_time
