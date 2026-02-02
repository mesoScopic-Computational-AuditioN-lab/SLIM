import warnings
import numpy as np
from tqdm.notebook import tqdm

from . import viz, utils

try:
    import torch
    TORCH_AVAILABLE = True
except:
    torch = None
    TORCH_AVAILABLE = False
    warnings.warn('torch installation not found! DNN Sensors will be disabled.', RuntimeWarning)

class SLIM(object):

    def __init__(self, dim, **kwargs):
        self.order = kwargs.get('order', 1)
        self.dim = dim
        self.verbose = kwargs.get('verbose', False)

        # maximum entropy initialization
        self.bel = np.array([np.ones(self.dim) / self.dim] * self.order)
        self.W = np.ones((self.dim ** self.order, self.dim)) / self.dim

        self.state_prediction_errors = [np.nan]
        self.context_weighted_error_potential = [np.nan]

        self.W_hist = [self.W]
        self.surprise_hist = [0]  # kl between prior and posterior
        self.y_hist = []

        self.x_hat_hist = [[np.nan] * self.dim] * self.order
        self.cv_hist = []

        self.track_dyn = kwargs.get('track_dyn', False)


        self.feedforward_from_levels = kwargs.get('feedforward_from_levels', None)
        self.feedback_from_levels = kwargs.get('feedback_from_levels', None)

        self.nle = []

        self.sensor = kwargs.get('sensor', None)

        self.model_name = kwargs.get('model_name', None)

        self.is_plt_set = False

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
            self.fit([state], lr, update_model_dyamics=False, leave=False)
            if isinstance(separator, list):
                self.fit([np.random.choice(separator)], lr, leave=False)
            else:
                self.fit([separator], lr, leave=False)
        self.reset()

    def reset(self):
        self.bel = np.array([np.ones(self.dim) / self.dim] * self.order)
        self.state_prediction_errors = [np.nan]
        self.context_weighted_error_potential = [np.nan]
        self.y_hist = []  # *self.order
        self.W_hist = [self.W]
        self.x_hat_hist = [[np.nan] * self.dim] * self.order
        self.cv_hist = []
        self.nle = []
        self.surprise_hist = [0]

    def flush_history(self):
        self.bel = np.array([self.bel[-1]])
        self.state_prediction_errors = self.state_prediction_errors[:-1]
        self.context_weighted_error_potential = self.context_weighted_error_potential[:-1]
        self.y_hist = self.y_hist[:-1]
        self.W_hist = [self.W]
        self.x_hat_hist = []
        self.cv_hist = []
        self.nle = []
        self.surprise_hist = []

    def fit(self, x, learning_rate=0.01, feed_observations=False, update_model_dyamics=True, leave=True, verbose=False):
        if self.sensor is None:
            raise ValueError('Must initialize sensor model before training (see SLIM(..., sensor=sensor).')

        loop = tqdm(range(len(x)), leave=leave, total=len(x), disable=not verbose)

        for t in loop:
            observation = self.observe(x[t], feed_observations)
            self.step(observation, learning_rate=learning_rate, update_model_dynamics=update_model_dyamics)


    def observe(self, state, feed_observations=False):
        if feed_observations:
            return state
        if isinstance(self.sensor, np.ndarray):
            return self.sensor[state]
        elif TORCH_AVAILABLE and isinstance(self.sensor, torch.nn.Module):
            state_tensor = torch.Tensor(state).unsqueeze(0).unsqueeze(0)
            obs = utils.softmax(self.sensor(state_tensor).detach().numpy().squeeze()).astype(np.float16)

            # hotfix for 5-dim states! (relevant for behavioral results)
            if self.dim == 5:
                f_mask = [0, 1, 3, 4, 5]
                logits = self.sensor(state_tensor).detach().numpy().squeeze()[f_mask]
                obs = utils.softmax(logits)
                return obs

            return obs
        else:
            raise ValueError('Could not determine sensor type...')

    def predict(self, fb_prediction=None):
        # short-term dynamics estimate
        W_stde = self.bel[-self.order]
        for idx in reversed(range(1, self.order)):
            W_stde = np.outer(W_stde, self.bel[-idx])

        # prediction
        x_hat = W_stde.flatten() @ self.W

        if fb_prediction is not None:  # this is when a lower level receives a prediction
            if self.verbose:
                print(f'\t recv: x_hat(level={self.feedback_from_levels})')
            x_hat = .7*x_hat + 0.3*fb_prediction

        self.x_hat_hist.append(x_hat)
        return x_hat

    def update(self, observation, context, lr, update_model_dynamics=None, scale_lr_by_sensor_uncertainty=None):

        # dynamics 'prediction error'
        post_mat = np.tile(self.bel[-1], reps=(self.dim ** self.order, 1))
        W_delta = context * (post_mat - self.W)

        self.context_weighted_error_potential.append(W_delta)

        if not update_model_dynamics:
            return
        if scale_lr_by_sensor_uncertainty:  # that is a hyper_parameter
            entropy = -np.sum(observation*np.log2(observation + 1e-10))
            factor = entropy**scale_lr_by_sensor_uncertainty
            self.W = self.W + factor * lr * W_delta
        else:
            self.W = self.W + lr * W_delta

    def step(self, observation, learning_rate=0.1, fb_prediction=None, update_model_dynamics=True, scale_lr_by_sensor_uncertainty=0):

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
        x_posterior = observation * x_hat
        x_posterior /= np.sum(x_posterior)

        # keep track of measures
        self.surprise_hist.append(utils.kl(x_hat, x_posterior))
        self.y_hist.append(observation)
        self.bel = np.vstack([self.bel, x_posterior])
        if self.track_dyn:
            self.cv_hist.append(context)
            self.W_hist.append(self.W)

        # update internal representation
        self.update(observation, context, learning_rate,
                    update_model_dynamics=update_model_dynamics,
                    scale_lr_by_sensor_uncertainty=scale_lr_by_sensor_uncertainty)

        return x_posterior

    # helpers for fast plotting

    def plot_belief_history(self, **kwargs):
        viz.plot_history(self.bel, **kwargs)