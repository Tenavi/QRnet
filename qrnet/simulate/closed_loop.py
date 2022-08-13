import numpy as np
from ._integrate import solve_ivp

def sim_closed_loop(
        dynamics, jacobian, controller, tspan, X0, t_eval=None, events=None,
        solver='RK45', atol=1e-06, rtol=1e-03
    ):
    '''
    Simulate the closed-loop system for a fixed time interval.

    Parameters
    ----------
        OCP: instance of TemplateOCP defining dynamics, Jacobian, etc.
        tspan: integration time, list of two floats
        X0: initial condition, (n,) numpy array
        controller: instance of a trained QRnet
        solver: ODE solver to use, str
        atol: absolute integration tolerance, float
        rtol: relative integration tolerance, float
        t_eval: optional (Nt,) numpy array of time instances to evaluate solution at

    Returns
    -------
        t: time vector, (Nt,) numpy array
        X: state time series, (n,Nt) numpy array
    '''
    def dynamics_wrapper(t, X):
        U = controller.eval_U(X)
        return dynamics(X, U)

    def jac_wrapper(t, X):
        return jacobian(X, controller)

    ode_sol = solve_ivp(
        dynamics_wrapper, tspan, X0, t_eval=t_eval, jac=jac_wrapper,
        events=events, vectorized=True, method=solver, rtol=rtol, atol=atol
    )

    return ode_sol.t, ode_sol.y, ode_sol.status

def sim_to_converge(
        dynamics, jacobian, controller, X0, config, events=None
    ):
    '''
    Simulate the closed-loop system until reach t_max or the dX/dt = 0.

    Parameters
    ----------
        OCP: instance of a setupProblem class defining dynamics, Jacobian, etc.
        config: a configuration dict defined in problem_def.py
        X0: initial condition, (n,) numpy array
        controller: instance of a trained QRnet

    Returns
    -------
        t: time vector, (Nt,) numpy array
        X: state time series, (n,Nt) numpy array
        converged: whether or not equilibrium was reached, bool
    '''

    t = np.zeros(1)
    X = X0.reshape(-1,1)

    converged = False

    # Solves over an extended time interval if needed to make ||f(x,u)|| -> 0
    while not converged and t[-1] < config.t1_max:
        t1 = np.maximum(config.t1_sim, t[-1] * config.t1_scale)
        # Simulate the closed-loop system
        t_new, X_new, status = sim_closed_loop(
            dynamics,
            jacobian,
            controller,
            [t[-1], t1],
            X[:,-1],
            events=events,
            solver=config.ode_solver,
            atol=config.atol,
            rtol=config.rtol
        )

        t = np.concatenate((t, t_new[1:]))
        X = np.hstack((X, X_new[:,1:]))

        if status == 1:
            break

        U = controller.eval_U(X[:,-1])
        converged = np.linalg.norm(dynamics(X[:,-1], U)) < config.fp_tol

    return t, X, converged
