"""Policy class for Task Embedding envs."""
import abc

import akro
import numpy as np

from garage.tf.policies import StochasticPolicy


class TaskEmbeddingPolicy(StochasticPolicy):
    """Base class for Task Embedding policies in TensorFlow.

    This policy needs a task id in addition to observation to sample an action.

    Args:
        name (str): Policy name, also the variable scope.
        env_spec (garage.envs.EnvSpec): Environment specification.
        encoder (garage.tf.embeddings.StochasticEncoder):
            A encoder that embeds a task id to a latent.

    """

    # pylint: disable=too-many-public-methods

    def __init__(self, name, env_spec, encoder):
        super().__init__(name, env_spec)
        self._encoder = encoder
        self._task_observation_space = self.concat_spaces(
            self._env_spec.observation_space, self.task_space)

    @property
    def encoder(self):
        """garage.tf.embeddings.encoder.Encoder: Encoder."""
        return self._encoder

    def get_latent(self, task_id):
        """Get embedded task id in latent space.

        Args:
            task_id (np.ndarray): One-hot task id.

        Returns:
            np.ndarray: An embedding sampled from embedding distribution.
            dict: Embedding distribution information.

        Note:
            It returns an embedding and a dict, with keys
            - mean (numpy.ndarray): Mean of the distribution.
            - log_std (numpy.ndarray): Log standard deviation of the
                distribution.

        """
        return self.encoder.forward(task_id)

    @property
    def latent_space(self):
        """akro.Box: Space of latent."""
        return self.encoder.spec.output_space

    @property
    def task_space(self):
        """akro.Box: One-hot space of task id."""
        return self.encoder.spec.input_space

    @property
    def task_observation_space(self):
        """akro.Box: Concatenated one-hot task id and observation space."""
        return self._task_observation_space

    @property
    def embedding_distribution(self):
        """garage.tf.distributions.DiagonalGaussian: Embedding distribution."""
        return self.encoder.distribution

    @abc.abstractmethod
    def get_action_under_task(self, observation, task_id):
        """Sample an action given observation and task id.

        Args:
            observation (np.ndarray): Observation from the environment.
            task_id (np.ndarry): One-hot task id.

        Returns:
            np.ndarray: Action sampled from the policy.

        """

    @abc.abstractmethod
    def get_actions_under_tasks(self, observations, task_ids):
        """Sample a batch of actions given observations and task ids.

        Args:
            observations (np.ndarray): Observations from the environment.
            task_ids (np.ndarry): One-hot task ids.

        Returns:
            np.ndarray: Actions sampled from the policy.

        """

    @abc.abstractmethod
    def get_action_under_latent(self, observation, latent):
        """Sample an action given observation and latent.

        Args:
            observation (np.ndarray): Observation from the environment.
            latent (np.ndarray): Latent.

        Returns:
            np.ndarray: Action sampled from the policy.

        """

    @abc.abstractmethod
    def get_actions_under_latents(self, observations, latents):
        """Sample a batch of actions given observations and latents.

        Args:
            observations (np.ndarray): Observations from the environment.
            latents (np.ndarray): Latents.

        Returns:
            np.ndarray: Actions sampled from the policy.

        """

    def embedding_dist_info_sym(self,
                                input_var,
                                state_info_vars=None,
                                name='embedding_dist_info_sym'):
        """Return the symbolic distribution information about the embedding.

        Args:
            input_var(tf.Tensor): Symbolic variable for encoder input.
            state_info_vars(dict): A dictionary whose values should contain
                information about the state of the policy at the time it
                receives the input.
            name (str): Name for symbolic graph.

        Returns:
            dict[tf.Tensor]: Outputs of the symbolic graph of distribution
                parameters.

        """
        return self.encoder.dist_info_sym(input_var, state_info_vars, name)

    def embedding_dist_info(self, input_val, state_infos=None):
        """Return the distribution information about the embedding.

        Args:
            input_val(tf.Tensor): Encoder input values.
            state_infos(dict): A dictionary whose values should contain
                information about the state of the policy at the time it
                receives the input.

        Returns:
            dict[numpy.ndarray]: Distribution parameters.

        """
        return self.encoder.dist_info(input_val, state_infos)

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    def get_action(self, observation):
        """Get action sampled from the policy.

        This function is not implemented because Task Embedding policy requires
        an additional task id to sample action.

        Args:
            observation (np.ndarray): Observation from the environment.

        Returns:
            np.ndarray: Action sampled from the policy.

        """
        raise NotImplementedError

    def get_actions(self, observations):
        """Get action sampled from the policy.

        This function is not implemented because Task Embedding policy requires
        an additional task id to sample action.

        Args:
            observations (list[np.ndarray]): Observations from the environment.

        Returns:
            np.ndarray: Actions sampled from the policy.

        """
        raise NotImplementedError

    def dist_info_sym(self, input_var, state_info_vars=None, name='default'):
        """Symbolic graph of action distribution.

        Return the symbolic distribution information about the actions.

        This function is not implemented because Task Embedding policy requires
        an additional task id to sample action.

        Args:
            input_var (tf.Tensor): symbolic variable for observations
            state_info_vars (dict): a dictionary whose values should contain
                information about the state of the policy at the time it
                received the observation.
            name (str): Name of the symbolic graph.

        """
        raise NotImplementedError

    def dist_info(self, input_val, state_infos):
        """Action distribution info.

        Return the distribution information about the actions.

        This function is not implemented because Task Embedding policy requires
        an additional task id to sample action.

        Args:
            input_val (tf.Tensor): observation values
            state_infos (dict): a dictionary whose values should contain
                information about the state of the policy at the time it
                received the observation

        """
        raise NotImplementedError

    def get_trainable_vars(self):
        """Get trainable variables.

        The trainable vars of a multitask policy should be the trainable vars
        of its model and the trainable vars of its embedding model.

        Returns:
            List[tf.Variable]: A list of trainable variables in the current
                variable scope.

        """
        return (self._variable_scope.trainable_variables() +
                self.encoder.get_trainable_vars())

    def get_global_vars(self):
        """Get global variables.

        The global vars of a multitask policy should be the global vars
        of its model and the trainable vars of its embedding model.

        Returns:
            List[tf.Variable]: A list of global variables in the current
                variable scope.

        """
        return (self._variable_scope.global_variables() +
                self.encoder.get_global_vars())

    def split_task_observation(self, collated):
        """Splits up observation into one-hot task and environment observation.

        Args:
            collated (np.ndarray): Environment observation concatenated with
                task one-hot.

        Returns:
            np.ndarray: Vanilla environment observation.
            np.ndarray: Task one-hot.

        """
        task_dim = self.task_space.flat_dim
        return collated[:-task_dim], collated[-task_dim:]

    @staticmethod
    def concat_spaces(first, second):
        """Concatenate two Box space.

        Args:
            first (akro.Box): The first space.
            second (akro.Box): The second space.

        Returns:
            akro.Box: The concatenated space.

        """
        assert isinstance(first, akro.Box)
        assert isinstance(second, akro.Box)

        first_lb, first_ub = first.bounds
        second_lb, second_ub = second.bounds
        first_lb, first_ub = first_lb.flatten(), first_ub.flatten()
        second_lb, second_ub = second_lb.flatten(), second_ub.flatten()
        return akro.Box(np.concatenate([first_lb, second_lb]),
                        np.concatenate([first_ub, second_ub]))
