import numpy as np
import tensorflow as tf

from .base import BaseNN
from .utilities import make_dense_graph, tf_jacobian
from .optimize import optimize

class GradientNN(BaseNN):
    '''
    A NN for approximating the gradient of the value function of an infinite
    horizon OCP.
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
        parameters : dict, optional
            Dict containing a list of pre-trained weights and biases for each
            layer under the key 'weights', arranged as
            [weights(layer1), biases(layer1), weights(layer2), biases(layer2), ...]
        '''
        if n_out is None:
            n_out = LQR.n_states

        super().__init__(
            LQR,
            n_hidden=n_hidden,
            n_neurons=n_neurons,
            n_out=n_out,
            activation=activation,
            U_star_fun=U_star_fun,
            scaling=scaling,
            parameters=parameters
        )

    def _setup_scaling(self, scaling=None, data=None):
        '''
        Setup input and output scaling parameters for the network. If scaling
        has been pre-computed elsewhere and supplied, uses this. Otherwise if
        data is supplied, computes scaling parameters appropriately.

        Parameters
        ----------
        var_names : list of strings
            Which variables to compute scaling parameters for.
        scaling : dict, optional, containing
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
        data : dict, optional, containing
            X : (n_states, n_data) array
                Input state data
            U : (n_controls, n_data) array
                Optimal control data
            dVdX : (n_states, n_data) array
                Value function gradient data

        Returns
        ----------
        scaling : dict
            Dictionary of arrays specifying scaling for inputs and outputs. May
            not have meaningful contents if success_flag is False
        success_flag : bool
            True if scaling was initialized successfully
        '''
        return super()._setup_scaling(
            ['X', 'U', 'dVdX'], scaling=scaling, data=data
        )

    def _build_graph(self):
        '''
        Build the computational graph for the NN and its loss functions.
        '''
        n_states = self.LQR.n_states
        n_controls = self.LQR.n_controls

        # Builds computational graph
        self.X_tf = tf.placeholder(tf.float32, shape=(n_states, None))
        self.dVdX_tf = tf.placeholder(tf.float32, shape=(n_states, None))
        self.U_tf = tf.placeholder(tf.float32, shape=(n_controls, None))

        self.dVdX_scaled_tf = tf.placeholder(tf.float32, shape=(n_states, None))
        self.U_scaled_tf = tf.placeholder(tf.float32, shape=(n_controls, None))

        dVdX_scaled_pred, self.dVdX_pred = self._make_eval_graph(self.X_tf)

        self.U_pred = self.U_star_fun(self.X_tf, self.dVdX_pred)
        self.dUdX = tf_jacobian(self.U_pred, self.X_tf)
        U_scaled_pred = self.U_scale*(self.U_pred - self.U_lb) - 1.

        # Value gradient loss using scaled data
        self.loss_dVdX = tf.reduce_mean(
            tf.reduce_sum((dVdX_scaled_pred - self.dVdX_scaled_tf)**2, axis=0)
        )

        # Control loss using scaled data
        self.loss_U = tf.reduce_mean(
            tf.reduce_sum((U_scaled_pred - self.U_scaled_tf)**2, axis=0)
        )

        self.gradient_loss_weight = tf.placeholder(tf.float32)

        self.loss = self.loss_U + self.gradient_loss_weight * self.loss_dVdX

    def _make_eval_graph(self, X):
        '''
        Helper function which builds a dense NN and transforms the output to
        make the prediction tensor operations.

        Arguments
        ----------
        X : (n_states, n_data) tensor
            State locations to make predictions for

        Returns
        ----------
        dVdX_scaled : (n_states, n_data) tensor
            Linearly scaled value gradient predictions for each state
        dVdX : (n_states, n_data) tensor
            Value gradient predictions for each state in original domain
        '''

        # Raw NN prediction in the scaled domain
        dVdX_scaled = make_dense_graph(
            X - self.LQR.X_bar,
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        dVdX = (dVdX_scaled + 1.)/self.dVdX_scale + self.dVdX_lb

        return dVdX_scaled, dVdX

    def bvp_guess(self, X):
        '''
        Predicts the value function V(X), its gradient dVdX(X), and the optimal
        control U(X) for each sample state in X. If the network does not make
        predictions for a quantity, then return the LQR approximation.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            State(s) to make predictions for.

        Returns
        ----------
        V : (1, n_data) or (1,) array
            Value function prediction for each column in X.
        dVdX : (n_states, n_data) or (n_states,) array
            Value gradient prediction for each column in X.
        U : (n_controls, n_data) or (n_controls,) array
            NN feedback control for each column in X.
        '''
        V, _, _ = self.LQR.bvp_guess(X)

        dVdX, U = self.sess.run(
            (self.dVdX_pred, self.U_pred),
            {self.X_tf: X.reshape(X.shape[0], -1)}
        )
        if X.ndim < 2:
            U = U.flatten()
        return V, dVdX.reshape(X.shape), U.astype(np.float64)

    def train(
            self, data, optimizer='L-BFGS-B', optimizer_opts={},
            batch_size=None, n_epochs=1, callback_epoch=1,
            gradient_loss_weight=1., **kwargs
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
        optimizer_opts : dict
            Dict of options to pass to the optimizer.
        batch_size : int, optional
            Number of data points (not trajectories) to use for SGD optimizers.
            If used in conjunction with 'L-BFGS-B', sets the maximum number of
            data to use.
        n_epochs : int, default=1
            How many times to iterate through the dataset (for SGD optimizers).
        callback_epoch : int, default=1
            Specifies after how many epochs to print loss functions (for SGD
            optimizers).
        gradient_loss_weight : float, default=1.
            Scalar multiplier in front of the value gradient mean square loss
            term.
        '''
        self._build(data=data)

        U_scaled = self.U_scale*(data['U'] - self.U_lb) - 1.
        dVdX_scaled = self.dVdX_scale*(data['dVdX'] - self.dVdX_lb) - 1.

        print('\nGradient loss weight = %1.1e' % gradient_loss_weight)

        def callback(feed_dict, session):
            print(
                '\nloss_U = %1.2e, loss_dVdX = %1.2e'
                % session.run((self.loss_U, self.loss_dVdX), feed_dict)
            )

        optimize(
            self.loss, self.sess,
            [self.X_tf, self.U_scaled_tf, self.dVdX_scaled_tf],
            [data['X'], U_scaled, dVdX_scaled],
            parameter_keys=[self.gradient_loss_weight],
            parameter_vals=[gradient_loss_weight],
            batch_size=batch_size,
            n_epochs=n_epochs,
            optimizer=optimizer,
            options=optimizer_opts,
            callback=callback,
            callback_epoch=callback_epoch
        )

# ---------------------------------------------------------------------------- #

class GradientQRnet(GradientNN):
    def _make_eval_graph(self, X):
        '''
        Helper function which builds a dense NN and transforms the output to
        make the prediction tensor operations.

        Arguments
        ----------
        X : (n_states, n_data) tensor
            State locations to make predictions for

        Returns
        ----------
        dVdX_scaled : (n_states, n_data) tensor
            Linearly scaled value gradient predictions for each state
        dVdX : (n_states, n_data) tensor
            Value gradient predictions for each state in original domain
        '''

        X_err = X - self.LQR.X_bar

        # Raw NN prediction in the scaled domain
        dVdX_scaled = make_dense_graph(
            X_err,
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        # Subtract NN contribution at zero
        dVdX_scaled_0 = make_dense_graph(
            tf.zeros((self.LQR.n_states, 1), dtype=tf.float32),
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        dVdX_scaled = dVdX_scaled - dVdX_scaled_0

        # LQR component
        PX = tf.matmul(2. * self.LQR.P.astype(np.float32), X_err)

        dVdX = PX + dVdX_scaled / self.dVdX_scale

        dVdX_scaled = dVdX_scaled + self.dVdX_scale*(PX - self.dVdX_lb) - 1.

        return dVdX_scaled, dVdX

# ---------------------------------------------------------------------------- #

class GradientMatQRnet(GradientNN):
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
        parameters : dict, optional
            Dict containing a list of pre-trained weights and biases for each
            layer under the key 'weights', arranged as
            [weights(layer1), biases(layer1), weights(layer2), biases(layer2), ...]
        '''
        super().__init__(
            LQR,
            n_hidden=n_hidden,
            n_neurons=n_neurons,
            n_out=LQR.n_states**2,
            activation=activation,
            U_star_fun=U_star_fun,
            scaling=scaling,
            parameters=parameters
        )

    def _make_eval_graph(self, X):
        '''
        Helper function which builds a dense NN and transforms the output to
        make the prediction tensor operations.

        Arguments
        ----------
        X : (n_states, n_data) tensor
            State locations to make predictions for

        Returns
        ----------
        dVdX_scaled : (n_states, n_data) tensor
            Linearly scaled value gradient predictions for each state
        dVdX : (n_states, n_data) tensor
            Value gradient predictions for each state in original domain
        '''

        n = self.LQR.n_states

        X_err = X - self.LQR.X_bar

        # Raw NN prediction in the scaled domain
        dVdX_scaled = make_dense_graph(
            X_err,
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        # NN contribution at zero
        dVdX_scaled_0 = make_dense_graph(
            tf.zeros((n, 1), dtype=tf.float32),
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        # Center NN prediction and reshape into matrix
        dVdX_scaled = tf.reshape(dVdX_scaled - dVdX_scaled_0, (n, n, -1))

        # Batch matrix multiplication of dVdX_mat and (X - Xf)
        dVdX_scaled = tf.einsum('ijk,jk->ik', dVdX_scaled, X_err)

        # LQR component
        PX = tf.matmul(2. * self.LQR.P.astype(np.float32), X_err)

        dVdX = PX + dVdX_scaled / self.dVdX_scale

        dVdX_scaled = dVdX_scaled + self.dVdX_scale*(PX - self.dVdX_lb) - 1.

        return dVdX_scaled, dVdX

# ---------------------------------------------------------------------------- #

class GradientJacQRnet(GradientNN):
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
        parameters : dict, optional
            Dict containing a list of pre-trained weights and biases for each
            layer under the key 'weights', arranged as
            [weights(layer1), biases(layer1), weights(layer2), biases(layer2), ...]
        '''
        super().__init__(
            LQR,
            n_hidden=n_hidden,
            n_neurons=n_neurons,
            activation=activation,
            U_star_fun=U_star_fun,
            scaling=scaling,
            parameters=parameters
        )

        if self.initialized_graph:
            self.dVdX_pred, self.U_pred, self.dUdX = self._freeze_Jacobian(
                self.X_tf
            )

    def _freeze_Jacobian(self, X):
        '''
        Evaluate the NN Jacobian at zero and create a tensorflow graph to
        compute the control and control Jacobian with this frozen numpy array.

        Arguments
        ----------
        X : (n_states, n_data) tensor
            State locations to make predictions for

        Returns
        ----------
        U : (n_controls, n_data) tensor
            Control predictions for each state
        dUdX : (n_controls, n_states, n_data) tensor
            Jacobian ofontrol predictions for each state
        '''
        zeros = tf.zeros((self.LQR.n_states, 1), dtype=tf.float32)

        dVdX_scaled_0 = make_dense_graph(
            zeros,
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        dVdX2_scaled = tf_jacobian(dVdX_scaled_0, zeros, stop_gradients=zeros)
        dVdX2_scaled = tf.squeeze(dVdX2_scaled, axis=-1)
        dVdX2_scaled = self.sess.run(dVdX2_scaled)

        _, dVdX_pred = self._make_eval_graph(X, dVdX2_scaled=dVdX2_scaled)

        U_pred = self.U_star_fun(X, dVdX_pred)
        dUdX = tf_jacobian(U_pred, X)

        return dVdX_pred, U_pred, dUdX

    def _make_eval_graph(self, X, dVdX2_scaled=None):
        '''
        Helper function which builds a dense NN and transforms the output to
        make the prediction tensor operations.

        Arguments
        ----------
        X : (n_states, n_data) tensor
            State locations to make predictions for
        dVdX2_scaled : (n_states, n_states) array, optional
            Fixed evaluation of the NN Jacobian at X=X_bar

        Returns
        ----------
        dVdX_scaled : (n_states, n_data) tensor
            Linearly scaled value gradient predictions for each state
        dVdX : (n_states, n_data) tensor
            Value gradient predictions for each state in original domain
        '''

        n = self.LQR.n_states

        X_err = X - self.LQR.X_bar
        zeros = tf.zeros((self.LQR.n_states, 1), dtype=tf.float32)

        # Raw NN prediction in the scaled domain
        dVdX_scaled = make_dense_graph(
            X_err,
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        # NN contribution at zero
        dVdX_scaled_0 = make_dense_graph(
            zeros,
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        if dVdX2_scaled is None:
            # Get Jacobian at origin
            dVdX2_scaled = tf_jacobian(dVdX_scaled_0, zeros, stop_gradients=zeros)
            dVdX2_scaled = tf.squeeze(dVdX2_scaled, axis=-1)

        # Matrix multiplication by (X - Xf)
        dVdX2_scaled = tf.matmul(dVdX2_scaled, X_err)

        dVdX_scaled = dVdX_scaled - dVdX_scaled_0 - dVdX2_scaled

        # LQR component
        PX = tf.matmul(2. * self.LQR.P.astype(np.float32), X_err)

        dVdX = PX + dVdX_scaled / self.dVdX_scale

        dVdX_scaled = dVdX_scaled + self.dVdX_scale*(PX - self.dVdX_lb) - 1.

        return dVdX_scaled, dVdX

    def train(self, data):
        '''
        Train the NN model given a data set.

        Arguments
        ----------
        data : dict containing
            X : (n_states, n_data) array
                Input state data
            U : (n_controls, n_data) array
                Optimal control data
            dVdX : (n_states, n_data) array
                Value function gradient data
        '''
        super().train(data)
        self.dVdX_pred, self.U_pred, self.dUdX = self._freeze_Jacobian(
            self.X_tf
        )
