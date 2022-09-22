# QRnet

This software repository is a proof of concept implementation of neural network (NN) optimal feedback controllers with local stability guarantees. See the associated papers for details:

  * [Neural Network Optimal Feedback Control with Guaranteed Local Stability](https://doi.org/10.1109/OJCSYS.2022.3205863)
  * [Neural Network Optimal Feedback Control with Enhanced Closed Loop Stability](https://doi.org/10.23919/ACC53348.2022.9867619)
  * [QRnet: Optimal Regulator Design With LQR-Augmented Neural Networks](https://doi.org/10.1109/LCSYS.2020.3034415)

If you use this software, please cite one or more of the above works. Please reach out with any questions, or if you encounter bugs or other problems.

## Installation

To install the `qrnet` package (in developer mode) run `pip install -e .` from the command line. This package has been developed and tested with the following software dependencies:

    python=3.6
    tensorflow-gpu=1.11
    scipy=1.5
    numpy=1.16
    pandas=1.1
    tqdm=4.50
    dill=0.3.2
    pylgr

The `pylgr` package can be downloaded at [https://github.com/Tenavi/PyLGR](https://github.com/Tenavi/PyLGR).

`qrnet` may work with other versions of the packages listed above, but this is not guaranteed. In the future we plan to update `qrnet` to use `Tensorflow 2`.

The examples included in this repo make use of some MATLAB code, which has been tested in version R2022a.

## The `qrnet` package

This is the main installable package which contains the working components for data generation, NN design, training, validation, and closed loop simulation.

* `qrnet.problem_template` contains base classes `MakeConfig` and `TemplateOCP` which should be subclassed and instantiated when defining a new optimal control problem (OCP).

* `qrnet.generate` contains the `generate` function to generate optimal control data by solving open loop optimal control problems, and the `refine` function to use previously generated data as an initial guess for a higher quality OCP solver (e.g. when starting with a direct method and then switching to an indirect method).

* `qrnet.validate` contains the `eval_errors` function which evaluate NN controller approximation errors, and `find_fixed_point` function to compute closed loop equilibria.

* `qrnet.controllers` is a module containing the different NN architectures, and an implementation of LQR in the same API.

    * To see the available architectures, call `available_models`.

    * To instantiate a new NN controller, first create an LQR controller with the  `LQR` class. Then call `create_NN`.

    * To load a previously trained NN controller, call `load_NN`.

* `qrnet.simulate` is a module implementing closed loop simulations and open loop OCP solvers.

    * `solve_ocp` is a function for solving open loop OCPs.

    * `sim_closed_loop` is a function which integrates a nonlinear dynamical system with NN or LQR controllers.

    * `sim_to_converge` is a wrapper function of `sim_closed_loop` which continues simulating the system until a steady state is reached or a specified event occurs.

    * `monte_carlo` is a wrapper function of `sim_to_converge` which conducts a set of closed loop simulations and optionally compares the results to open loop optimal solutions.

## Running example scripts

#### `problem.py`

This script is used by scripts in the `run` folder. It imports `config` and `OCP` instances from `examples` depending on which example OCP is selected (not commented out). It also defines and creates some needed directories for saving data, models, and simulation results.

### The `run` folder

These are standalone scripts which exemplify different functionalities of the `qrnet` package. The scripts in this folder make use of `problem.py` in the root directory, described above. They are meant to be run from the root directory and accept command line arguments. Common command line arguments are:

* `-rs` (`int`, default=`None`): Random seed to use for sampling initial conditions and initializing NN weights.

* `-mt` (`int`, default=`None`): Integer specifying which previously trained NN to load. Either a 10 digit timestamp, or a python list index referring to the model in order of oldest to most recent.

* `-dt` ({`"train"`, `"test"`}, default=`"train"`) : Command line argument specifying a data type to generate.

* `-xd` (`float`, default=`None`): Optionally specify the norm of initial conditions used in closed loop simulations.

* `-io`: If present when running (Monte Carlo) closed loop simulations, instead of randomly sampling a new initial condition, import (random) open loop OCP solution(s) from test data and run the simulation from this initial condition.

* `-v` (`int`, default=`0`): Level of verbosity used for some scripts. See each script for options.

##### The `run` folder is organized as follows:

* `run.data` shows how to use `qrnet.generate` to generate open loop optimal control data. Data sets are saved as `.mat` files which can be accessed in MATLAB and python with the `scipy.io` module.

    * `run.data.generate` generates training or test data. Accepts `rs`, `mt`, `dt`, and `v` arguments. If `mt` is given, use NN warm start, otherwise use LQR warm start.

    * `run.data.refine` takes existing data as an initial guess and runs it through another open loop OCP solver. A typical use case is to start by specifying `ocp_solver="direct"` in the `MakeConfig`, then running `run.data.refine` after setting `ocp_solver="indirect"`. Accepts `dt` and `v` arguments.

* `run.train` shows how to load a data set generated using the scripts in `run.data` and train a NN controller using this data. Models are saved as pickle files and information about each model is added to `model_info.csv`, a spreadsheet in the same directory with the saved model. Accepts `a`, `rs`, and `mt` arguments. If `mt` is specified, loads a previously trained model and continues to train it with whatever training data is now available. The `a` argument is a string which specifies the model architecture to use (run `qrnet.controllers.available_models` for a list of options). Run with `-a LQR` to save a baseline LQR model for comparison.

* `run.predict` uses a trained NN to make predictions for the optimal control, value function, and/or value gradient at points in the training or test data set. Accepts `dt` and `mt` arguments.

* `run.simulate` shows how to use the `qrnet.simulate` module for (Monte Carlo) closed loop simulations. Results are saved as `.mat` files which can be accessed in MATLAB and python with the `scipy.io` module.

    * `run.simulate.closed_loop` simulates the system under both LQR and NN control, and attempts to solve the open loop OCP for comparison. Accepts `mt`, `rs`, `xd`, and `io` arguments.

    * `run.simulate.monte_carlo` runs NN-in-the-loop Monte Carlo simulations using `qrnet.simulate.monte_carlo`. Accepts `mt`, `rs`, `xd`, `io`, `o`, `v`, and `r` arguments. If `mt` is given, runs Monte Carlo simulations for only this model timestamp, otherwise runs simulations for **all** NNs which do not have existing Monte Carlo results. The `o` argument takes no parameters, run with `-o` to solve the open loop OCP for each initial condition in the Monte Carlo simulation. The `r` argument takes no parameters, run with `-r` to run Monte Carlo simulations in reverse order starting from the newest NN and ending with the oldest.

    * `run.simulate.LQR_only` simulates the system under LQR control. This is useful for testing a system and LQR gains prior to generating data and training NN models. Accepts `rs` and `xd` arguments.

    * `run.simulate.uncontrolled` simulates the system with a single constant control input. May be useful for testing if dynamics are correctly implemented. Accepts `rs` and `xd` arguments.

* `run.clean` is a script which deletes saved NNs and Monte Carlo results for models which have been deleted from `model_info.csv`.

### The `examples` folder

This folder contains the example OCPs presented in the referenced papers, and some additional unpublished examples. Critically, each control problem is itself treated as a module which defines `config`, an instance of `qrnet.problem_template.MakeConfig`, and `MakeOCP`, a subclass of `qrnet.problem_template.TemplateOCP` which is later instantiated in `problem.py`. These can be found and adjusted in `examples.<example>.problem_def.py`. Each example has several MATLAB scripts which can be used to plot simulation results. The example problems are

* `examples.van_der_pol`: Stabilization of the Van der Pol oscillator with bounded control input. This is the simplest OCP and should be tried first to understand the workflow in the `run` folder and test the code. Some small data sets and example models are provided for reference.

* `examples.satellite`: Quaternion-based attitude control of a nonlinear rigid body satellite with torque saturation.

* `examples.burgers`: Stabilization of discretized Burgers equation with destabilizing reaction term and two distributed control inputs. The dimension of the discretization can be specified by changing `examples.burgers.problem_def.N_STATES`.

* `examples.uav`: Stabilization, altitude tracking, and course tracking for a six degree of freedom fixed wing unmanned aircraft. This is a more realistic - and accordingly more challenging - example than the others. Some aspects of this example problem are not fully implemented.

The `examples` folder also contains `examples.plotting`, a folder of MATLAB scripts used to visualize training and Monte Carlo results. These scripts were used to generate figures in the listed references. The user will probably want to write their own plotting scripts; these are provided only for reference.

* `examples.plotting.plot_bar` plots statistics from `model_info.csv` and Monte Carlo tests using bar graphs. The x axis is the number of trajectories used for training, so each bar represents a collection of NNs of particular type trained with a certain number of trajectories.

* `examples.plotting.plot_scatter` plot statistics from `model_info.csv` and Monte Carlo tests using scatter plots. The x axis can be anything, such as the mean relative control error. Each point on the plot represents a single NN.

* `examples.plotting.plot_settings` is a script which contains common configurations to both `plot_bar` and `plot_scatter`, such as the control problem to plot results for and which control architectures to show on the plots.
