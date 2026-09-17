"""
model.py

Defines the SLIM model.
"""

import warnings
import numpy as np
from tqdm.notebook import tqdm

from . import utils

try:
    import torch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    TORCH_AVAILABLE = False
    warnings.warn('torch installation not found! DNN Sensors will be disabled.', RuntimeWarning)

class SLIM(object):
    """
    Represents an SLIM model object.
    """

    def __init__(self, dim, **kwargs):
        """
        Constructor of a SLIM instance. It requires dimensionality to be specified.
        Args:
            dim: Dimensionality of the state space.
            **kwargs: Optional arguments include:
                * sensor (np.array or torch model): this specifies the sensor compartment and should usually
                 be specified here. It can be ignored in cases where observations are directly fed into SLIM.
                * order (int): specifies the hierarchical order of the SLIM model.
                * verbose (bool): displays information such as fitting iteration and predictions.
                * track_dyn (bool): whether to track the transition matrices of the SLIM model.
                * model_name (str): gives the specific SLIM instance a name.

        This constructor creates internal variables for posterior (bel) and transition matrix (W).
        """
        self.dim = dim
        self.sensor = kwargs.get('sensor', None)
        self.order = kwargs.get('order', 1)
        self.verbose = kwargs.get('verbose', False)
        self.track_dyn = kwargs.get('track_dyn', False)
        self.model_name = kwargs.get('model_name', None)

        # maximum entropy initialization
        self.bel = np.array([np.ones(self.dim) / self.dim] * self.order)
        self.W = np.ones((self.dim ** self.order, self.dim)) / self.dim

        self.state_prediction_errors = [np.nan]
        self.context_weighted_error_potential = [np.nan]
        self.W_hist = [self.W]
        self.bayesian_surprise_hist = [0]
        self.y_hist = []
        self.x_hat_hist = [[np.nan] * self.dim] * self.order
        self.cv_hist = []


    def __repr__(self):
        return (f'SLIM(\n'
                f'\torder={self.order},\n'
                f'\tdim={self.dim},\n'
                f'\ttrack_dyn={self.track_dyn},\n'
                # f'\tW=\n{np.round(self.W, 2)},\n'
                # f'\th=\n{np.round(self.h_measurement, 2)},\n'
                f')')


    def warm_up(self, states, epochs, lr=0.01, separator=0):
        """
        Warm up the internal representation of the model to learn about states in silence.
        :param lr: learning rate
        :param states: internal representation of the model
        :param separator: separation between states
        :param epochs: number of epochs
        :return:
        """

        # create sequence and inject separator
        for _ in range(epochs):
            state = np.random.choice(states)
            self.fit([state], lr, update_model_dynamics=False, leave=False)
            if isinstance(separator, list):
                self.fit([np.random.choice(separator)], lr, leave=False)
            else:
                self.fit([separator], lr, leave=False)
        self.reset()

    def reset(self):
        """
        Reset the internal representation of the model to its initial state.

        """
        self.bel = np.array([np.ones(self.dim) / self.dim] * self.order)
        self.W = np.ones((self.dim ** self.order, self.dim)) / self.dim
        self.state_prediction_errors = [np.nan]
        self.context_weighted_error_potential = [np.nan]
        self.y_hist = []  # *self.order
        self.W_hist = [self.W]
        self.x_hat_hist = [[np.nan] * self.dim] * self.order
        self.cv_hist = []
        self.bayesian_surprise_hist = [0]

    def flush_history(self):
        """
        Clears history without resetting.

        """
        self.bel = np.array([self.bel[-1]])
        self.state_prediction_errors = self.state_prediction_errors[:-1]
        self.context_weighted_error_potential = self.context_weighted_error_potential[:-1]
        self.y_hist = self.y_hist[:-1]
        self.W_hist = [self.W]
        self.x_hat_hist = []
        self.cv_hist = []
        self.bayesian_surprise_hist = []

    def fit(self, x, learning_rate=0.01, feed_observations=False, update_model_dynamics=True, leave=True):
        """
        Feeds states to SLIM by iteratively calling the step function.

        :param x: sequence of states.
        :param learning_rate: The learning rate.
        :param feed_observations: whether to bypass the sensor.
        :param update_model_dynamics: whether to update the transition matrix.
        :param leave: whether to keep the update loop (verbosity has to be set for this).
        :return: returns the SLIM instance
        """
        if self.sensor is None and not feed_observations:
            raise ValueError('Must initialize sensor model before training (see SLIM(..., sensor=sensor).')

        loop = tqdm(range(len(x)), leave=leave, total=len(x), disable=not self.verbose)

        for t in loop:
            observation = self.observe(x[t], feed_observations)
            self.step(observation, learning_rate=learning_rate, update_model_dynamics=update_model_dynamics)

        return self


    def observe(self, state, feed_observations=False):
        """
        Applies the sensor to the state.
        Args:
            state: a state value.
            feed_observations: views the state variable as an observation and skips the sensor call.

        Returns:
            obs: returns the observation.

        """
        if feed_observations:
            return state
        if isinstance(self.sensor, np.ndarray):
            return self.sensor[state]
        elif TORCH_AVAILABLE and isinstance(self.sensor, torch.nn.Module):
            state_tensor = torch.Tensor(state).unsqueeze(0).unsqueeze(0)
            obs = utils.softmax(self.sensor(state_tensor).detach().numpy().squeeze()).astype(np.float16)

            # hotfix for 5-dim states! (relevant for behavioral modeling results)
            if self.dim == 5:
                f_mask = [0, 1, 3, 4, 5]
                logits = self.sensor(state_tensor).detach().numpy().squeeze()[f_mask]
                obs = utils.softmax(logits)
                return obs

            return obs
        else:
            raise ValueError('Could not determine sensor type...')

    def predict(self, fb_prediction=None):
        """
        Predicts the current state by integrating the transition matrix with previous beliefs.
        Args:
            fb_prediction (np.array): feedback from other models/hierarchical levels.

        Returns:
            x_hat (np.array): vector of probability masses for the predicted current state.
        """

        W_stde = self.bel[-self.order]  # short-term dynamics estimate.
        for idx in reversed(range(1, self.order)):
            W_stde = np.outer(W_stde, self.bel[-idx])

        # prediction
        x_hat = W_stde.flatten() @ self.W

        if fb_prediction is not None:  # this is when a lower level receives a prediction
            x_hat = .7*x_hat + 0.3*fb_prediction

        self.x_hat_hist.append(x_hat)
        return x_hat

    def update(self, context, lr, update_model_dynamics=None):
        """
        Updates the transition matrix.
        Args:
            context: (history) of past beliefs.
            lr: learning rate.
            update_model_dynamics: whether to update the transition matrix.

        """
        # dynamics 'prediction error'
        post_mat = np.tile(self.bel[-1], reps=(self.dim ** self.order, 1))
        W_delta = context * (post_mat - self.W)

        self.context_weighted_error_potential.append(W_delta)

        if not update_model_dynamics:
            return

        self.W = self.W + lr * W_delta

    def step(self, observation, learning_rate=0.1, fb_prediction=None, update_model_dynamics=True, skip_inference=False):
        """
        The step function defines the main functionality of SLIM.
        1. predicts the current state using the previous belief and transition matrix.
        2. calculates a state prediction error
        3. composes the context vector
        4. calculates the posterior
        Args:
            observation (np.array): a vector representing the likelihood.
            learning_rate (float): learning rate (ideally between 0 an 1).
            fb_prediction (np.array): feedback from other models/hierarchical levels.
            update_model_dynamics (bool): whether to update the transition matrix.
            skip_inference (bool): whether to perform inference or operate on observations.

        Returns:
            x_posterior: returns the posterior belief about x
        """
        if observation is None:
            print(f'\t Warning, model <{self.model_name}> got no measurement, initializing with a uniform distribution.')
            observation = np.ones(self.dim) / self.dim

        # buffering routine
        if len(self.y_hist) < self.order:
            self.y_hist.append(observation)
            if self.verbose:
                print('\t sent: None (buffering)')
            return None

        x_hat = self.predict(fb_prediction=fb_prediction)

        # prediction error (based on state likelihood from sensor model)
        epsilon = np.linalg.norm(x_hat - observation)
        self.state_prediction_errors.append(float(epsilon))

        # context
        context = self.bel[-self.order]
        for idx in reversed(range(1, self.order)):
            context = np.outer(context, self.bel[-idx])
        # ignore this for now
        if isinstance(context, int):
            context = np.array([context])
        context = np.tile(context.flatten(), reps=(self.dim, 1)).T

        # inference
        if not skip_inference:
            x_posterior = observation * x_hat
            x_posterior /= np.sum(x_posterior)
        else:
            x_posterior = observation

        # keep track of measures
        self.bayesian_surprise_hist.append(utils.kl(x_posterior, x_hat))
        self.y_hist.append(observation)
        self.bel = np.vstack([self.bel, x_posterior])
        if self.track_dyn:
            self.cv_hist.append(context)
            self.W_hist.append(self.W)

        # update internal representation
        self.update(context, learning_rate, update_model_dynamics=update_model_dynamics)

        return x_posterior
