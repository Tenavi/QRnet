import os
import time
import dill
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

from .utilities import initialize_dense, make_dense_graph

class BaseController:
    '''
    Base class for implementing a state feedback controller.
    '''
    def __init__(self, *args, **kwargs):
        pass

    def architecture(self):
        return type(self).__name__

    def _get_learned_params(self):
        '''
        Get numpy values for the trainable tensorflow variables such as NN
        weights. By default returns the NN weights but should be overwritten to
        return other variables as needed.

        Returns
        -------
        parameters : dict
            Dictionary of (lists of) arrays giving trained parameter values
        '''
        return {}

    def eval_V(self, X):
        '''
        Predicts the value function, V(X), for each sample state in X.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to predict the value function for.

        Returns
        -------
        dVdX : (1, n_data) or (1,) array
            Value function prediction for each column in X.
        '''
        X_err = X.reshape(X.shape[0], -1) - self.X_bar
        XPX = X_err * np.matmul(self.P, X_err)
        XPX = np.sum(XPX, axis=0, keepdims=True)

        if X.ndim < 2:
            XPX = XPX.flatten()

        return XPX

    def eval_dVdX(self, X):
        '''
        Predicts the value function gradient, dV/dX(X), for each sample state in
        X.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to predict the value gradient for.

        Returns
        -------
        dVdX : (n_states, n_data) or (n_states,) array
            Value gradient prediction for each column in X.
        '''
        raise NotImplementedError

    def eval_U(self, X):
        '''
        Evaluates the NN feedback control, U(X), for each sample state in X.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to evaluate the control for.

        Returns
        -------
        U : (n_controls, n_data) or (n_controls,) array
            NN feedback control for each column in X.
        '''
        raise NotImplementedError

    def eval_dUdX(self, X):
        '''
        Evaluates the Jacobian of the NN feedback control, [dU/dX](X), for each
        sample state in X.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to evaluate the control for.

        Returns
        -------
        dUdX : (n_controls, n_states, n_data) or (n_controls, n_states) array
            Jacobian of NN feedback control for each column in X.
        '''
        raise NotImplementedError

    def bvp_guess(self, X, eval_U=False):
        '''
        Predicts the value function V(X), its gradient dVdX(X), and the optimal
        control U(X) for each sample state in X. If the network does not make
        predictions for a quantity, then return the LQR approximation.

        Parameters
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to make predictions for.
        eval_U : bool, default=False
            Specify as True to return the control prediction U(X), otherwise
            just computes the value function and gradient predictions

        Returns
        -------
        V : (1, n_data) or (1,) array
            Value function prediction for each column in X.
        dVdX : (n_states, n_data) or (n_states,) array
            Value gradient prediction for each column in X.
        U : (n_controls, n_data) or (n_controls,) array
            NN feedback control for each column in X, only returned if
            eval_U=True.
        '''
        raise NotImplementedError

    def train(
            self, data,
            optimizer='L-BFGS-B', optimizer_opts={},
            batch_size=None, n_epochs=1,
            gradient_loss_weight=1., value_loss_weight=1., **kwargs
        ):
        '''
        Optimize the training loss using limited memory BFGS.

        Parameters
        ----------
        data : dict
            Dict of open loop optimal control data containing
            X : (n_states, n_data) array
                Input state data
            U : (n_controls, n_data) array
                Optimal control data
            dVdX : (n_states, n_data) array
                Value function gradient data
            V : (1, n_data) array
                Value function data
        optimizer : str, default='L-BFGS-B'
            Which optimizer to use. Options are 'L-BFGS-B' and any optimizer
            implemented in tensorflow.train.
        optimizer_opts : dict
            Dict of options to pass to the optimizer.
        batch_size : int, optional
            Number of data points (not trajectories) to use for SGD optimizers.
            If used in conjunction with 'L-BFGS-B', sets the maximum number of
            data to use.
        n_epochs : int, default=1
            How many times to iterate through the dataset (for SGD optimizers).
        gradient_loss_weight : float, default=1.
            Scalar multiplier in front of the value gradient mean square loss
            term. Not used by control_networks.
        value_loss_weight : float, default=1.
            Scalar multiplier in front of the value function mean square loss
            term. Not used by control_networks and gradient_networks.
        '''
        raise NotImplementedError

    def save(self, model_dir, error_dict, random_seed):
        '''
        Saves the model parameters to a pickle file. Also stores error data and
        important configuration parameters in a .csv file.

        Parameters
        ----------
        model_dir : path-like
            Which folder to save model parameters and information in
        error_dict : dict
            Dictionary containing error metrics to save
        random_seed : int or None
            Random seed set prior to training
        '''
        timestamp = int(time.time())

        self._save_parameters(timestamp, model_dir)
        self._save_info(timestamp, model_dir, error_dict, random_seed)

    def _save_parameters(self, timestamp, model_dir):
        '''
        Saves the model parameters to a pickle file in the specified directory.

        Parameters
        ----------
        timestamp : int
            UTC time at which the model was saved, without milliseconds
        model_dir : path-like
            Which folder to save model parameters and information in
        '''

        model_dict = {
            'architecture': self.architecture(),
            'parameters': self._get_learned_params()
        }
        for param in ['LQR', 'U_star_fun', 'activation', 'scaling']:
            if hasattr(self, param):
                model_dict[param] = getattr(self, param)
        if self.architecture() == 'LQR':
            model_dict['LQR'] = self

        timestamp = str(timestamp)

        model_path = os.path.join(model_dir, timestamp + '.pkl')

        with open(model_path, 'wb') as model_file:
            dill.dump(model_dict, model_file)

        print('Model saved with timestamp ' + timestamp)

    def _save_info(self, timestamp, model_dir, error_dict, random_seed):
        '''
        Stores error data and important configuration parameters to the
        model_info.csv file in the specified directory.

        Parameters
        ----------
        timestamp : int
            UTC time at which the model was saved, without milliseconds
        model_dir : path-like
            Which folder to save model parameters and information in
        error_dict : dict
            Dictionary containing error metrics to save
        random_seed : int or None
            Random seed set prior to training
        '''
        current_time = datetime.fromtimestamp(timestamp)
        timestamp = str(timestamp)

        model_info = {
            **error_dict,
            'architecture': self.architecture(),
            'random_seed': random_seed,
            'timestamp': timestamp,
            'year': current_time.year,
            'month': current_time.month,
            'day': current_time.day,
            'hour': current_time.hour,
            'minute': current_time.minute,
        }
        if hasattr(self, 'activation'):
            model_info['activation'] = self.activation

        # Pandas likes making dataframes from dicts where everything is a list
        for key in model_info.keys():
            model_info[key] = [model_info[key]]

        info_path = os.path.join(model_dir, 'model_info.csv')

        try:
            model_info_df = pd.read_csv(info_path, index_col=0)
            model_info_df = model_info_df.append(
                pd.DataFrame.from_dict(model_info)
            )
            model_info_df = model_info_df.reset_index(drop=True)
        except:
            model_info_df = pd.DataFrame.from_dict(model_info)

        model_info_df.to_csv(info_path)

class BaseNN(BaseController):
    '''
    Base class for implementing a NN for approximating the optimal feedback
    control of an infinite horizon OCP.
    '''
    def __init__(
            self, LQR, n_hidden=None, n_neurons=None, n_out=None,
            activation='tanh', U_star_fun=None, scaling={}, parameters={}
        ):
        '''
        Build the computational graph for the NN and its loss functions. If
        scaling has been pre-computed elsewhere and supplied, uses this.
        Otherwise if data is supplied, computes scaling parameters
        appropriately. Initializes NN parameters.

        Parameters
        ----------
        LQR : object
            Instance of controllers.linear_quadratic_regulator.LQR.
        n_hidden : int
            Number of hidden layers. Required if weights is None.
        n_neurons : int
            Number of neurons per hidden layer. Required if weights is None.
        n_out : int
            Number of output neurons. Required if weights is None.
        activation : str, default='tanh'
            Activation function to use for hidden layers.
        U_star_fun : callable
            Function which evaluates the optimal control based on the state and
            value gradient. Takes two tensor arguments and outputs a tensor.
            Required for value_networks and gradient_networks.
        scaling : dict, optional
            Dict specifying scaling for inputs and outputs containing
            X_lb : (n_states, 1) array
                Lower bound of input data
            X_ub : (n_states, 1) array
                Upper bound of input data
            U_lb : (n_controls, 1) array
                Lower bound of control data
            U_ub : (n_controls, 1) array
                Upper bound of control data
            dVdX_lb : (n_states, 1) array
                Lower bound of gradient/costate data
            dVdX_ub : (n_states, 1) array
                Upper bound of gradient/costate data
            V_ub : float
                Upper bound of value function data
        parameters : dict, optional
            Dict containing a list of pre-trained weights and biases for each
            layer under the key 'weights', arranged as
            [weights(layer1), biases(layer1), weights(layer2), biases(layer2), ...]
        '''
        self.LQR = LQR
        self.activation = activation
        self.U_star_fun = U_star_fun

        self.weights = initialize_dense(
            n_in=LQR.n_states,
            n_out=n_out,
            n_hidden=n_hidden,
            n_neurons=n_neurons,
            weights=parameters.get('weights')
        )

        self.initialized_graph = False
        self._build(scaling=scaling)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build(self, scaling={}, data={}):
        '''
        Build the computational graph for the NN and its loss functions. If
        scaling has been pre-computed elsewhere and supplied, uses this.
        Otherwise if data is supplied, computes scaling parameters
        appropriately.

        Parameters
        ----------
        scaling : dict, optional
            Dict of arrays specifying scaling for inputs and outputs containing
            X_lb : (n_states, 1) array
                Lower bound of input data
            X_ub : (n_states, 1) array
                Upper bound of input data
            U_lb : (n_controls, 1) array
                Lower bound of control data
            U_ub : (n_controls, 1) array
                Upper bound of control data
            dVdX_lb : (n_states, 1) array
                Lower bound of gradient/costate data
            dVdX_ub : (n_states, 1) array
                Upper bound of gradient/costate data
            V_ub : float
                Upper bound of value function data
        data : dict, optional
            Dict of open loop optimal control data containing
            X : (n_states, n_data) array
                Input state data
            U : (n_controls, n_data) array
                Optimal control data
            dVdX : (n_states, n_data) array
                Value function gradient data
            V : (1, n_data) array
                Value function data
        '''
        if not self.initialized_graph:
            self.scaling, success_flag = self._setup_scaling(
                scaling=scaling, data=data
            )
            if success_flag:
                self._build_graph()
                self.initialized_graph = True

    def _build_graph(self):
        '''
        Build the computational graph for the NN and its loss functions. Should
        be implemented by all subclasses.
        '''
        raise NotImplementedError

    def _setup_scaling(self, var_names, scaling={}, data={}):
        '''
        Setup input and output scaling parameters for the network. If scaling
        has been pre-computed elsewhere and supplied, uses this. Otherwise if
        data is supplied, computes scaling parameters appropriately.

        Parameters
        ----------
        var_names : list of strings
            Which variables to compute scaling parameters for.
        scaling : dict, optional
            Dict of arrays specifying scaling for inputs and outputs containing
            X_lb : (n_states, 1) array
                Lower bound of input data
            X_ub : (n_states, 1) array
                Upper bound of input data
            U_lb : (n_controls, 1) array
                Lower bound of control data
            U_ub : (n_controls, 1) array
                Upper bound of control data
            dVdX_lb : (n_states, 1) array
                Lower bound of gradient/costate data
            dVdX_ub : (n_states, 1) array
                Upper bound of gradient/costate data
            V_ub : float
                Upper bound of value function data
        data : dict, optional
            Dict of open loop optimal control data containing
            X : (n_states, n_data) array
                Input state data
            U : (n_controls, n_data) array
                Optimal control data
            dVdX : (n_states, n_data) array
                Value function gradient data
            V : (1, n_data) array
                Value function data

        Returns
        -------
        scaling : dict
            Dictionary of arrays specifying scaling for inputs and outputs. May
            not have meaningful contents if success_flag is False
        success_flag : bool
            True if scaling was initialized successfully
        '''
        for bound in ['U_lb', 'U_ub']:
            if bound not in scaling and getattr(self.LQR, bound) is not None:
                scaling[bound] = getattr(self.LQR, bound)

        scaling['V_lb'] = 0.

        scaling_funs = {'lb': np.min, 'ub': np.max}

        # Loop over variables in data
        for var in var_names:
            # Loop over scaling functions to compute
            for fun_name, fun in scaling_funs.items():
                key = var + '_' + fun_name
                # If haven't already computed this parameter, compute it now
                if key not in scaling and var in data:
                    scaling[key] = fun(data[var], axis=1, keepdims=True)

        # Check to make sure all combinations of variables and scaling functions
        # were computed
        success_flag = all([
            '_'.join(pair) in scaling
            for pair in itertools.product(var_names, scaling_funs)
        ])

        if success_flag:
            if 'X_lb' in scaling and 'X_ub' in scaling:
                self.X_scale = 2./(scaling['X_ub'] - scaling['X_lb'])

            if 'U_lb' in scaling and 'U_ub' in scaling:
                self.U_lb = scaling['U_lb']
                self.U_ub = scaling['U_ub']
                self.U_scale = 2./(self.U_ub - self.U_lb)

            if 'dVdX_lb' in scaling and 'dVdX_ub' in scaling:
                self.dVdX_lb = scaling['dVdX_lb']
                self.dVdX_ub = scaling['dVdX_ub']
                self.dVdX_scale = 2./(self.dVdX_ub - self.dVdX_lb)

            if 'V_ub' in scaling:
                self.V_ub = scaling['V_ub']

        return scaling, success_flag

    def _get_learned_params(self):
        '''
        Get numpy values for the trainable tensorflow variables such as NN
        weights. By default returns the NN weights but should be overwritten to
        return other variables as needed.

        Returns
        ----------
        parameters : dict
            Dictionary of (lists of) arrays giving trained parameter values
        '''
        return {'weights': self.sess.run(self.weights)}

    def eval_V(self, X):
        '''
        Predicts the value function, V(X), for each sample state in X.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to predict the value function for.

        Returns
        ----------
        dVdX : (1, n_data) or (1,) array
            Value function prediction for each column in X.
        '''
        if not hasattr(self, 'V_pred'):
            raise NotImplementedError

        V = self.sess.run(self.V_pred, {self.X_tf: X.reshape(X.shape[0], -1)})
        if X.ndim < 2:
            V = V.flatten()
        return V

    def eval_dVdX(self, X):
        '''
        Predicts the value function gradient, dV/dX(X), for each sample state in
        X.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to predict the value gradient for.

        Returns
        ----------
        dVdX : (n_states, n_data) or (n_states,) array
            Value gradient prediction for each column in X.
        '''
        if not hasattr(self, 'dVdX_pred'):
            raise NotImplementedError

        dVdX = self.sess.run(
            self.dVdX_pred, {self.X_tf: X.reshape(X.shape[0], -1)}
        )
        return dVdX.reshape(X.shape)

    def eval_U(self, X):
        '''
        Evaluates the NN feedback control, U(X), for each sample state in X.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to evaluate the control for.

        Returns
        ----------
        U : (n_controls, n_data) or (n_controls,) array
            NN feedback control for each column in X.
        '''
        U = self.sess.run(self.U_pred, {self.X_tf: X.reshape(X.shape[0], -1)})
        if X.ndim < 2:
            U = U.flatten()
        return U.astype(np.float64)

    def eval_dUdX(self, X):
        '''
        Evaluates the Jacobian of the NN feedback control, [dU/dX](X), for each
        sample state in X.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to evaluate the control for.

        Returns
        ----------
        dUdX : (n_controls, n_states, n_data) or (n_controls, n_states) array
            Jacobian of NN feedback control for each column in X.
        '''
        dUdX = self.sess.run(self.dUdX, {self.X_tf: X.reshape(X.shape[0], -1)})
        if X.ndim < 2:
            dUdX = dUdX[:,:,0]
        return dUdX.astype(np.float64)
