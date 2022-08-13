import numpy as np
import tensorflow as tf

from .base import BaseNN
from .utilities import make_dense_graph, tf_jacobian
from .optimize import optimize

from qrnet.utilities import saturate_tf

class ControlNN(BaseNN):
    '''
    A NN for approximating the optimal feedback control of an infinite
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
        U_star_fun : ignored
            Not used, for API consistency only.
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
        parameters : dict, optional
            Dict containing a list of pre-trained weights and biases for each
            layer under the key 'weights', arranged as
            [weights(layer1), biases(layer1), weights(layer2), biases(layer2), ...]
        '''
        if n_out is None:
            n_out = LQR.n_controls

        super().__init__(
            LQR,
            n_hidden=n_hidden,
            n_neurons=n_neurons,
            n_out=n_out,
            activation=activation,
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
        scaling : dict, optional, containing
            X_lb : (n_states, 1) array
                Lower bound of input data
            X_ub : (n_states, 1) array
                Upper bound of input data
            U_lb : (n_controls, 1) array
                Lower bound of control data
            U_ub : (n_controls, 1) array
                Upper bound of control data
        data : dict, optional, containing
            X : (n_states, n_data) array
                Input state data
            U : (n_controls, n_data) array
                Optimal control data

        Returns
        ----------
        scaling : dict
            Dictionary of arrays specifying scaling for inputs and outputs. May
            not have meaningful contents if success_flag is False
        success_flag : bool
            True if scaling was initialized successfully
        '''
        return super()._setup_scaling(['X', 'U'], scaling=scaling, data=data)

    def _build_graph(self):
        '''
        Build the computational graph for the NN and its loss functions.
        '''
        n_states = self.LQR.n_states
        n_controls = self.LQR.n_controls

        # Builds computational graph
        self.X_tf = tf.placeholder(tf.float32, shape=(n_states, None))
        self.U_tf = tf.placeholder(tf.float32, shape=(n_controls, None))
        self.U_scaled_tf = tf.placeholder(tf.float32, shape=(n_controls, None))

        U_scaled_pred, self.U_pred = self._make_eval_graph(self.X_tf)

        self.dUdX = tf_jacobian(self.U_pred, self.X_tf)

        # Control loss using scaled data
        self.loss_U = tf.reduce_mean(
            tf.reduce_sum((U_scaled_pred - self.U_scaled_tf)**2, axis=0)
        )

        self.loss = self.loss_U

    def _saturate_smooth(self, U, U_scaled=None):
        '''
        Smoothed control saturation using generalized logistic function.
        Parameters c1, c2 are calculated to preserve local behavior, i.e.
        saturation function has unit derivative, near equilibrium Uf.

        Arguments
        ----------
        U : (n_controls, None) tensor
            Control tensor to saturate in the unscaled (original) domain
        U_scaled : (n_controls, None) tensor, optional
            Control tensor in the scaled domain, gets passed back unchanged if
            no saturation was performed

        Returns
        ----------
        U : (n_controls, None) tensor
            Controls saturated between U_lb and U_ub
        U_scaled : (n_controls, None) tensor, or same as U_scaled input
            Controls saturated between -1 and +1, or the unchanged input
        '''
        U_lb, U_ub, Uf = self.LQR.U_lb, self.LQR.U_ub, self.LQR.U_bar

        if U_lb is not None and U_ub is not None:
            c1 = (U_ub - Uf)/(Uf - U_lb)
            c2 = (U_ub - U_lb)/((U_ub - Uf)*(Uf - U_lb))

            # Find the point at which the logistic function is numerically zero
            # This calculation can have problems if c1 is very small, but is
            # mostly fine since it is performed in double precision for single
            # precision-representable values
            eps = np.finfo(np.float32).resolution.astype(np.float64)
            exp_max = np.log((1. - eps)/eps) - np.log(c1)
            # Clip the value of the exponent to avoid problems with the gradient
            pwr = tf.clip_by_value(c2*(Uf - U), -exp_max, exp_max)

            U = U_lb + (U_ub - U_lb)/(1. + c1*tf.exp(pwr))

            U_scaled = (2./(U_ub - U_lb)) * (U - U_lb) - 1.
        elif U_lb is None and U_ub is None:
            pass
        else:
            raise NotImplementedError

        return U, U_scaled

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
        U_scaled : (n_controls, n_data) tensor
            Linearly scaled control predictions for each state
        U : (n_control, n_data) tensor
            Control predictions for each state in original domain
        '''

        # Raw NN prediction in the scaled domain
        U_scaled = make_dense_graph(
            X - self.LQR.X_bar,
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        U = (U_scaled + 1.)/self.U_scale + self.U_lb

        U, U_scaled = self._saturate_smooth(U, U_scaled=U_scaled)

        return U_scaled, U

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
        V, dVdX, _ = self.LQR.bvp_guess(X)

        U = self.sess.run(
            self.U_pred, {self.X_tf: X.reshape(X.shape[0], -1)}
        )
        if X.ndim < 2:
            U = U.flatten()
        return V, dVdX, U.astype(np.float64)

    def train(
            self, data, optimizer='L-BFGS-B', optimizer_opts={},
            batch_size=None, n_epochs=1, callback_epoch=1, **kwargs
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
        callback_epoch : int, default=1
            Specifies after how many epochs to print loss functions (for SGD
            optimizers).
        '''
        self._build(data=data)

        U_scaled = self.U_scale*(data['U'] - self.U_lb) - 1.

        def callback(feed_dict, session):
            print('\nloss_U = %1.2e' % session.run(self.loss_U, feed_dict))

        optimize(
            self.loss, self.sess,
            [self.X_tf, self.U_scaled_tf],
            [data['X'], U_scaled],
            batch_size=batch_size,
            n_epochs=n_epochs,
            optimizer=optimizer,
            options=optimizer_opts,
            callback=callback,
            callback_epoch=callback_epoch
        )

# ---------------------------------------------------------------------------- #

class ControlQRnet(ControlNN):
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
        U_scaled : (n_controls, n_data) tensor
            Linearly scaled control predictions for each state
        U : (n_control, n_data) tensor
            Control predictions for each state in original domain
        '''

        X_err = X - self.LQR.X_bar

        # Raw NN prediction in the scaled domain
        U_scaled = make_dense_graph(
            X_err,
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        # Subtract NN contribution at zero
        U_scaled_0 = make_dense_graph(
            tf.zeros((self.LQR.n_states, 1), dtype=tf.float32),
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        U_scaled = U_scaled - U_scaled_0

        # LQR component
        KX = self.LQR.U_bar - tf.matmul(self.LQR.K.astype(np.float32), X_err)
        KX = saturate_tf(KX, self.U_lb, self.U_ub)

        U = KX + U_scaled / self.U_scale

        # Default unsaturated U_scaled to use
        U_scaled = U_scaled + self.U_scale*(KX - self.U_lb) - 1.

        U, U_scaled = self._saturate_smooth(U, U_scaled=U_scaled)

        return U_scaled, U

# ---------------------------------------------------------------------------- #

class ControlMatQRnet(ControlNN):
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
        U_star_fun : ignored
            Not used, for API consistency only.
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
        parameters : dict, optional
            Dict containing a list of pre-trained weights and biases for each
            layer under the key 'weights', arranged as
            [weights(layer1), biases(layer1), weights(layer2), biases(layer2), ...]
        '''
        super().__init__(
            LQR,
            n_hidden=n_hidden,
            n_neurons=n_neurons,
            n_out=LQR.n_controls*LQR.n_states,
            activation=activation,
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
        U_scaled : (n_controls, n_data) tensor
            Linearly scaled control predictions for each state
        U : (n_control, n_data) tensor
            Control predictions for each state in original domain
        '''

        n = self.LQR.n_states
        m = self.LQR.n_controls

        X_err = X - self.LQR.X_bar

        # Raw NN prediction in the scaled domain
        U_scaled = make_dense_graph(
            X_err,
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        # NN contribution at zero
        U_scaled_0 = make_dense_graph(
            tf.zeros((n, 1), dtype=tf.float32),
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        # Center NN prediction and reshape into matrix
        U_scaled = tf.reshape(U_scaled - U_scaled_0, (m, n, -1))

        # Batch matrix multiplication of U_mat and (X - Xf)
        U_scaled = tf.einsum('ijk,jk->ik', U_scaled, X_err)

        # LQR component
        KX = self.LQR.U_bar - tf.matmul(self.LQR.K.astype(np.float32), X_err)
        KX = saturate_tf(KX, self.U_lb, self.U_ub)

        U = KX + U_scaled / self.U_scale

        # Default unsaturated U_scaled to use
        U_scaled = U_scaled + self.U_scale*(KX - self.U_lb) - 1.

        U, U_scaled = self._saturate_smooth(U, U_scaled=U_scaled)

        return U_scaled, U

# ---------------------------------------------------------------------------- #

class ControlJacQRnet(ControlNN):
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
        U_star_fun : ignored
            Not used, for API consistency only.
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
            scaling=scaling,
            parameters=parameters
        )

        if self.initialized_graph:
            self.U_pred, self.dUdX = self._freeze_Jacobian(self.X_tf)

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

        U_scaled_0 = make_dense_graph(
            zeros,
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        dUdX_scaled = tf_jacobian(U_scaled_0, zeros, stop_gradients=zeros)
        dUdX_scaled = tf.squeeze(dUdX_scaled, axis=-1)
        dUdX_scaled = self.sess.run(dUdX_scaled)

        _, U_pred = self._make_eval_graph(X, dUdX_scaled=dUdX_scaled)

        dUdX = tf_jacobian(U_pred, X)

        return U_pred, dUdX

    def _make_eval_graph(self, X, dUdX_scaled=None):
        '''
        Helper function which builds a dense NN and transforms the output to
        make the prediction tensor operations.

        Arguments
        ----------
        X : (n_states, n_data) tensor
            State locations to make predictions for
        dUdX_scaled : (n_controls, n_states) array, optional
            Fixed evaluation of the NN Jacobian at X=X_bar

        Returns
        ----------
        U_scaled : (n_controls, n_data) tensor
            Linearly scaled control predictions for each state
        U : (n_control, n_data) tensor
            Control predictions for each state in original domain
        '''

        n = self.LQR.n_states
        m = self.LQR.n_controls

        X_err = X - self.LQR.X_bar
        zeros = tf.zeros((n, 1), dtype=tf.float32)

        # Raw NN prediction in the scaled domain
        U_scaled = make_dense_graph(
            X_err,
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        # NN contribution at zero
        U_scaled_0 = make_dense_graph(
            zeros,
            self.X_scale,
            weights=self.weights,
            activation=self.activation
        )

        if dUdX_scaled is None:
            # Get Jacobian at origin
            dUdX_scaled = tf_jacobian(U_scaled_0, zeros, stop_gradients=zeros)
            dUdX_scaled = tf.squeeze(dUdX_scaled, axis=-1)

        # Matrix multiplication by (X - Xf)
        dUdX_scaled = tf.matmul(dUdX_scaled, X_err)

        U_scaled = U_scaled - U_scaled_0 - dUdX_scaled

        # LQR component
        KX = self.LQR.U_bar - tf.matmul(self.LQR.K.astype(np.float32), X_err)
        KX = saturate_tf(KX, self.U_lb, self.U_ub)

        U = KX + U_scaled / self.U_scale

        # Default unsaturated U_scaled to use
        U_scaled = U_scaled + self.U_scale*(KX - self.U_lb) - 1.

        U, U_scaled = self._saturate_smooth(U, U_scaled=U_scaled)

        return U_scaled, U

    def train(self, data, **kwargs):
        '''
        Train the NN model given a data set.

        Parameters
        ----------
        data : dict containing
            X : (n_states, n_data) array
                Input state data
            U : (n_controls, n_data) array
                Optimal control data
        '''
        super().train(data, **kwargs)
        self.U_pred, self.dUdX = self._freeze_Jacobian(self.X_tf)
