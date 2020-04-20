"""GaussianMLPTaskEmbeddingPolicy."""
import akro
import numpy as np
import tensorflow as tf

from garage.tf.models import GaussianMLPModel
from garage.tf.policies.task_embedding_policy import TaskEmbeddingPolicy


class GaussianMLPTaskEmbeddingPolicy(TaskEmbeddingPolicy):
    """GaussianMLPTaskEmbeddingPolicy.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        encoder (garage.tf.embeddings.StochasticEncoder): Embedding network.
        name (str): Model name, also the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor.
        learn_std (bool): Is std trainable.
        adaptive_std (bool): Is std a neural network. If False, it will be a
            parameter.
        std_share_network (bool): Boolean for whether mean and std share
            the same network.
        init_std (float): Initial value for std.
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues.
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues.
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        std_output_nonlinearity (callable): Nonlinearity for output layer in
            the std network. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        std_parameterization (str): How the std should be parametrized. There
            are a few options:
            - exp: the logarithm of the std will be stored, and applied a
                exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 env_spec,
                 encoder,
                 name='GaussianMLPTaskEmbeddingPolicy',
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.initializers.glorot_uniform(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(),
                 output_b_init=tf.zeros_initializer(),
                 learn_std=True,
                 adaptive_std=False,
                 std_share_network=False,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=tf.nn.tanh,
                 std_output_nonlinearity=None,
                 std_parameterization='exp',
                 layer_normalization=False):
        assert isinstance(env_spec.action_space, akro.Box)
        super().__init__(name, env_spec, encoder)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        self.model = GaussianMLPModel(
            output_dim=self._action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            adaptive_std=adaptive_std,
            std_share_network=std_share_network,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_hidden_sizes=std_hidden_sizes,
            std_hidden_nonlinearity=std_hidden_nonlinearity,
            std_output_nonlinearity=std_output_nonlinearity,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization,
            name='GaussianMLPModel')

        self._initialize()

    def _initialize(self):
        obs_input = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, self._obs_dim))
        # task_input = self._encoder.input
        latent_input = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self._encoder.output_dim))

        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs

            with tf.compat.v1.variable_scope('concat_obs_latent'):
                obs_latent_input = tf.concat([obs_input, latent_input],
                                             axis=-1)
            self.model.build(obs_latent_input, name='from_latent')

        self._f_dist_obs_latent = tf.compat.v1.get_default_session(
        ).make_callable([
            self.model.networks['from_latent'].mean,
            self.model.networks['from_latent'].log_std
        ],
                        feed_list=[obs_input, latent_input])

    def dist_info_sym_under_task(self,
                                 obs_var,
                                 task_var,
                                 state_info_vars=None,
                                 name='default'):
        """Build a symbolic graph of the action distribution given task.

        Args:
            obs_var (tf.Tensor): Symbolic observation input.
            task_var (tf.Tensor): Symbolic task input.
            state_info_vars (dict): Extra state information, e.g.
                previous action.
            name (str): Name for symbolic graph.

        Returns:
            dict[tf.Tensor]: Outputs of the symbolic graph of
                action distribution parameters.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            latent_dist_info_sym = self._encoder.dist_info_sym(task_var,
                                                               name=name)
            latent_var = self._encoder.distribution.sample_sym(
                latent_dist_info_sym)
            obs_latent_input = tf.concat([obs_var, latent_var], axis=-1)
            mean_var, log_std_var, _, _ = self.model.build(obs_latent_input,
                                                           name=name)
        return dict(mean=mean_var, log_std=log_std_var)

    def dist_info_sym_under_latent(self,
                                   obs_var,
                                   latent_var,
                                   state_info_vars=None,
                                   name='from_latent'):
        """Build a symbolic graph of the action distribution given latent.

        Args:
            obs_var (tf.Tensor): Symbolic observation input.
            latent_var (tf.Tensor): Symbolic latent input.
            state_info_vars (dict): Extra state information, e.g.
                previous action.
            name (str): Name for symbolic graph.

        Returns:
            dict[tf.Tensor]: Outputs of the symbolic graph of distribution
                parameters.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            obs_latent_input = tf.concat([obs_var, latent_var], axis=-1)
            mean_var, log_std_var, _, _ = self.model.build(obs_latent_input,
                                                           name=name)
        return dict(mean=mean_var, log_std=log_std_var)

    @property
    def distribution(self):
        """Policy action distribution.

        Returns:
            garage.tf.distributions.DiagonalGaussian: Policy distribution.

        """
        return self.model.networks['from_latent'].dist

    def get_action_under_latent(self, observation, latent):
        """Sample an action given observation and latent.

        Args:
            observation (np.ndarray): Observation from the environment.
            latent (np.ndarray): Latent.

        Returns:
            np.ndarray: Action sampled from the policy.

        """
        flat_obs = self.observation_space.flatten(observation)
        flat_latent = self.latent_space.flatten(latent)

        mean, log_std = self._f_dist_obs_latent([flat_obs], [flat_latent])
        rnd = np.random.normal(size=mean.shape)
        sample = rnd * np.exp(log_std) + mean
        sample = self.action_space.unflatten(sample[0])
        mean = self.action_space.unflatten(mean[0])
        log_std = self.action_space.unflatten(log_std[0])
        return sample, dict(mean=mean, log_std=log_std)

    def get_actions_under_latents(self, observations, latents):
        """Sample a batch of actions given observations and latents.

        Args:
            observations (np.ndarray): Observations from the environment.
            latents (np.ndarray): Latents.

        Returns:
            np.ndarray: Actions sampled from the policy.

        """
        raise NotImplementedError

    def get_action_under_task(self, observation, task_id):
        """Sample an action given observation and task id.

        Args:
            observation (np.ndarray): Observation from the environment.
            task_id (np.ndarry): One-hot task id.

        Returns:
            np.ndarray: Action sampled from the policy.

        """
        raise NotImplementedError

    def get_actions_under_tasks(self, observations, task_ids):
        """Sample a batch of actions given observations and task ids.

        Args:
            observations (np.ndarray): Observations from the environment.
            task_ids (np.ndarry): One-hot task ids.

        Returns:
            np.ndarray: Actions sampled from the policy.

        """
        raise NotImplementedError

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_dist_obs_latent']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize()
