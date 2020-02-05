#!/usr/bin/env python3
"""
This is an example to train a task with PPO algorithm.

Here it creates InvertedDoublePendulum using gym. And uses a PPO with 1M
steps.

Results:
    AverageDiscountedReturn: 500
    RiseTime: itr 40

"""
import random

import tensorflow as tf
from garage.envs import normalize, normalize_reward
from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from garage.experiment.deterministic import set_seed
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy
from metaworld.envs.mujoco.env_dict import EASY_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import EASY_MODE_CLS_DICT
from garage import wrap_experiment

MT10_envs_by_id = {
    task: env(*EASY_MODE_ARGS_KWARGS[task]['args'],
              **EASY_MODE_ARGS_KWARGS[task]['kwargs'])
    for (task, env) in EASY_MODE_CLS_DICT.items()
}

env_ids = ['reach-v1', 'push-v1', 'pick-place-v1', 'door-v1', 'drawer-open-v1', 'drawer-close-v1', 'button-press-topdown-v1', 'ped-insert-side-v1', 'window-open-v1', 'window-close-v1']
# env_ids = ['push-v1']
# env_ids = ['reach-v1']
# env_ids = ['pick-place-v1']

MT10_envs = [TfEnv(normalize_reward(MT10_envs_by_id[i])) for i in env_ids]

@wrap_experiment
def ppo_mt10(ctxt=None, seed=1):

    """Run task."""
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        env = MultiEnvWrapper(MT10_envs, env_ids, sample_strategy=round_robin_strategy)

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=None,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                hidden_sizes=(64, 64),
                use_trust_region=True,
            ),
        )

        algo = PPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=150,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
                tf_optimizer_args=dict(
                    learning_rate=3e-4,
                ),
            ),
            stop_entropy_gradient=True,
            entropy_method='max',
            policy_ent_coeff=0.002,
            center_adv=False,
        )

        runner.setup(algo, env)
        runner.train(n_epochs=1500, batch_size=len(MT10_envs)*10*150, plot=False)


seeds = random.sample(range(100), 1)
for seed in seeds:
    ppo_mt10(seed=seed)