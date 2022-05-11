## QRnet

See the associated papers at

  * [https://doi.org/10.48550/arXiv.2205.00394](https://doi.org/10.48550/arXiv.2205.00394)
  * [https://doi.org/10.48550/arXiv.2109.07466](https://doi.org/10.48550/arXiv.2109.07466)
  * [https://doi.org/10.1109/LCSYS.2020.3034415](https://doi.org/10.1109/LCSYS.2020.3034415)

##### Software recommendations:

tensorflow-gpu=1.11, scipy=1.5, numpy=1.16, pandas=1.1, tqdm=4.50, dill=0.3.2, pylgr

The pylgr package can be obtained at https://github.com/Tenavi/PyLGR.

How this repository is organized:

### run:

These are standalone scripts which exemplify different functionalities of the 'qrnet' module. They are meant to be run from the root directory and accept command line arguments. The scripts in this folder make use of problem.py in the root directory, which imports 'config' and 'OCP' instances from 'examples'.

  * run.generate.py: Generate open-loop optimal control data using 'qrnet.generate' and LQR or NN warm-start.

  * run.train.py: Train a neural network controller once data has been generated. Models are saved as pickle files and information about each model is added to a 'model_info.csv', a spreadsheet in the same directory. Run with 'python run.train.py -a LQR' to "train" a baseline LQR model for comparison. Models can subsequently be loaded using 'qrnet.load_NN'.

  * run.simulate.py: Simulate the neural network-in-the-loop system.

  * run.monte_carlo.py: Evaluate previously-trained neural network controllers on a large set of closed-loop simulations and compare to corresponding open-loop optimal solutions.

  * run.predict_grid.py: Evaluate the neural network control, value gradient, and value function predictions on a 2d spatial mesh.

  * run.clean.py: Clear out saved neural networks and monte carlo results for models which have been deleted from 'model_info.csv'.

### problem.py:

This script is used by scripts in the 'run' folder. It imports 'config' and 'OCP' instances from 'examples' depending on which example optimal control problem is selected (not commented out). It also defines and creates some needed directories for saving data, models, and simulation results.

### examples:

This module contains python and matlab files for the example optimal control problems in the referenced papers. Critically, each control problem is itself treated as a module which defines 'config', an instance of 'qrnet.problem_template.MakeConfig', and 'OCP', an instance of a subclass of 'qrnet.problem_template.TemplateOCP'. These can be found and adjusted in 'examples.<example>.problem_def.py'. The example problems are

 * examples.van_der_pol: Stabilization of the Van der Pol oscillator with bounded control input.

 * examples.burgers: Stabilization of discretized Burgers equation with destabilizing reaction term and two distributed control inputs.

 * examples.satellite: Quaternion-based attitude control of a nonlinear rigid body with torque saturation.

##### examples.plotting:

This folder also contains a folder of matlab functions which can be used to visualize the contents of 'model_info.csv' and the results of monte_carlo simulations.

 * examples.plotting.plot_bar.m: Plot statistics from 'model_info.csv' and monte carlo tests using bar graphs. The x axis is the number of trajectories used for training, so each bar represents a collection of neural networks of particular type trained with a certain number of trajectories.

 * examples.plotting.plot_scatter.m: Plot statistics from 'model_info.csv' and monte carlo tests using scatter plots. The x axis can be anything, such as the mean relative control error. Each point on the plot represents a single neural network.

 * examples.plotting.plot_settings.m: Change values in here to change configurations which are common to both 'plot_bar.m' and 'plot_scatter.m', such as the control problem to plot results for and which control architectures to show on the plots.

### qrnet:

This is the main installable package which contains the working components for data generation, neural network design, training, and validation, and closed-loop simulation.

* qrnet.problem_template: Contains base classes 'MakeConfig' and 'TemplateOCP' which should be subclassed and instantiated when defining a new optimal control problem.

* qrnet.generate: Functions to generate optimal control data by solving open-loop optimal control problems.

* qrnet.validate: Functions to evaluate neural network controller approximation errors and find closed-loop equilibria.

##### qrnet.controllers:

Module containing the different neural network architectures, as well as a model factory and 'load_NN' function to assist with model setup.

##### qrnet.simulate:

Module implementing closed-loop simulation and open-loop optimal control problem solving with neural network and LQR controllers.
