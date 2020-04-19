"""This modules creates a MTSAC model in PyTorch."""

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage import log_performance
from garage.torch.algos import SAC


class MTAC(SAC):
    """A MTSAC Model in Torch.





    Args:
        policy(garage.torch.policy): Policy/Actor/Agent that is being optimized
            by MTSAC.
        qf1(garage.torch.q_function): QFunction/Critic used for actor/policy
            optimization. See Soft Actor-Critic and Applications.
        qf2(garage.torch.q_function): QFunction/Critic used for actor/policy
            optimization. See Soft Actor-Critic and Applications.
        replay_buffer(garage.replay_buffer): Stores transitions that
            are previously collected by the sampler.
        env_spec(garage.envs.env_spec.EnvSpec): The env_spec attribute of the
            environment that the agent is being trained in. Usually accessable
            by calling env.spec.
        num_tasks(int): The number of tasks being learned.
        max_path_length(int): Max path length of the environment.
        gradient_steps_per_itr(int): Number of optimization steps that should
            occur before the training step is over and a new batch of
            transitions is collected by the sampler.
        use_automatic_entropy_tuning(bool): True if the entropy/temperature
            coefficient should be learned. False if it should be static.
        alpha(float): entropy/temperature to be used if
            `use_automatic_entropy_tuning` is False.
        target_entropy(float): target entropy to be used during
            entropy/temperature optimization. If None, the default heuristic
            from Soft Actor-Critic Algorithms and Applications is used.
        initial_log_entropy(float): initial entropy/temperature coefficient
            to be used if use_automatic_entropy_tuning is True.
        discount(float): Discount factor to be used during sampling and
            critic/q_function optimization.
        buffer_batch_size(int): The number of transitions sampled from the
            replay buffer that are used during a single optimization step.
        min_buffer_size(int): The minimum number of transitions that need to be
            in the replay buffer before training can begin.
        target_update_tau(float): coefficient that controls the rate at which
            the target q_functions update over optimization iterations.
        policy_lr(float): learning rate for policy optimizers.
        qf_lr(float): learning rate for q_function optimizers.
        reward_scale (float): reward scale. Changing this hyperparameter
            changes the effect that the reward from a transition will have
            during optimization.
        optimizer(torch.optim): optimizer to be used for policy/actor,
            q_functions/critics, and temperature/entropy optimizations.
        steps_per_epoch (int): Number of train_once calls per epoch.
        num_evaluation_trajs(int): The number of evaluation trajectories
            used for computing eval stats at the end of every epoch.

    """

    def __init__(
            self,
            policy,
            qf1,
            qf2,
            replay_buffer,
            env_spec,
            num_tasks,
            max_path_length,
            gradient_steps_per_itr,
            use_automatic_entropy_tuning=True,
            alpha=None,
            target_entropy=None,
            initial_log_entropy=0.,
            discount=0.99,
            buffer_batch_size=64,
            min_buffer_size=int(1e4),
            target_update_tau=5e-3,
            policy_lr=3e-4,
            qf_lr=3e-4,
            reward_scale=1.0,
            optimizer=torch.optim.Adam,
            steps_per_epoch=1,
            num_evaluation_trajs=5,
    ):

        super().__init__(
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            replay_buffer=replay_buffer,
            env_spec=env_spec,
            max_path_length=max_path_length,
            gradient_steps_per_itr=gradient_steps_per_itr,
            use_automatic_entropy_tuning=use_automatic_entropy_tuning,
            alpha=alpha,
            target_entropy=target_entropy,
            initial_log_entropy=initial_log_entropy,
            discount=discount,
            buffer_batch_size=buffer_batch_size,
            min_buffer_size=min_buffer_size,
            target_update_tau=target_update_tau,
            policy_lr=policy_lr,
            qf_lr=qf_lr,
            reward_scale=reward_scale,
            optimizer=optimizer,
            steps_per_epoch=steps_per_epoch,
            num_evaluation_trajs=num_evaluation_trajs)
        self._num_tasks = num_tasks
        if self.use_automatic_entropy_tuning and not alpha:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(
                    self.env_spec.action_space.shape).item()
            self.log_alpha = torch.Tensor([self._initial_log_entropy] *
                                          self._num_tasks).requires_grad_()
            self.alpha_optimizer = optimizer([self.log_alpha] *
                                             self._num_tasks,
                                             lr=self.policy_lr)
        else:
            self.log_alpha = torch.Tensor([alpha]).log()

    def _get_log_alpha(self, **kwargs):
        """Return the value of log_alpha.

        This function exists in case there are versions of MTSAC that need
        access to a modified log_alpha, such as multi_task MTSAC.

        Args:
            kwargs(dict): keyword args that can be used in retrieving the
                log_alpha parameter. Unused here.

        Returns:
            torch.Tensor: log_alpha

        """
        obs = kwargs['obs']
        log_alpha = self.log_alpha
        one_hots = obs[:, :self._num_tasks]
        ret = torch.mm(one_hots, log_alpha.unsqueeze(0).t()).squeeze()
        assert ret.size() == torch.Size([self.buffer_batch_size])
        return ret

    def optimize_policy(self, itr, samples_data):
        """Optimize the policy q_functions, and temperature coefficient.

        Args:
            itr (int): Iterations.
            samples_data (list): Processed batch data.

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        obs = samples_data['observation']
        qf1_loss, qf2_loss = self._critic_objective(
            samples_data, get_log_alpha_kwargs=dict(obs=obs))

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        action_dists = self.policy(obs)
        new_actions_pre_tanh, new_actions = (
            action_dists.rsample_with_pre_tanh_value())
        log_pi_new_actions = action_dists.log_prob(
            value=new_actions, pre_tanh_value=new_actions_pre_tanh)

        policy_loss = self._actor_objective(obs,
                                            new_actions,
                                            log_pi_new_actions,
                                            get_log_alpha_kwargs=dict(obs=obs))
        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        self.policy_optimizer.step()

        if self.use_automatic_entropy_tuning:
            alpha_loss = self._temperature_objective(
                log_pi_new_actions, get_log_alpha_kwargs=dict(obs=obs))
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        return policy_loss, qf1_loss, qf2_loss

    def _evaluate_policy(self, epoch, eval_env):
        """Evaluate the performance of the policy via deterministic rollouts.

            Statistics such as (average) discounted return and success rate are
            recorded.
        Args:
            epoch(int): The current training epoch.
            eval_env(garage.envs.GarageEnv): Environment that is used for
                evaluation of the policy.

        Returns:
            float: The average return across self._num_evaluation_trajs
                trajectories
        """
        epoch_local_success_rate = []
        for env in eval_env:
            _, avg_success_rate = log_performance(
                epoch,
                self._obtain_evaluation_samples(
                    env,
                    num_trajs=self._num_evaluation_trajs,
                    max_path_length=self.max_path_length),
                discount=self.discount,
                prefix=name)

            epoch_local_success_rate.append(avg_success_rate)
            assert len(epoch_local_success_rate) == self._num_tasks
            self.epoch_mean_success_rate.append(
                np.mean(epoch_local_success_rate))
            self.epoch_median_success_rate.append(
                np.median(epoch_local_success_rate))

            tabular.record('local/Mean_SuccessRate',
                           self.epoch_mean_success_rate[-1])
            tabular.record('local/Median_SuccessRate',
                           self.epoch_median_success_rate[-1])
            tabular.record('local/Max_Median_SuccessRate',
                           np.max(self.epoch_median_success_rate))
            tabular.record('local/Max_Mean_SuccessRate',
                           np.max(self.epoch_mean_success_rate))
        return last_return

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        super().to(device)
        self.log_alpha = torch.Tensor(
            [self._initial_log_entropy] *
            self._num_tasks).to(device).requires_grad_()
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer = self._optimizer([self.log_alpha],
                                                   lr=self.policy_lr)
