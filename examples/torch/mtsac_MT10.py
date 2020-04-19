import click
import numpy as np

from torch.nn import functional as F
from torch import nn as nn

from garage import wrap_experiment
from metaworld.benchmarks import MT10
from garage.envs import GarageEnv
from garage.experiment import LocalRunner, run_experiment
from garage.replay_buffer import SimpleReplayBuffer
from garage.torch.algos import MTSAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
import argparse
from garage.sampler import SimpleSampler
import garage.torch.utils as tu


@click.command()
@click.option('--gpu', '_gpu', type=int, default=0)
@wrap_experiment(snapshot_mode='none')
def mt10_sac_normalize_reward(ctxt=None, seed=1):
    """Set up environment and algorithm and run the task."""
    runner = LocalRunner(ctxt)
    MT10_envs_by_id = {}
    MT10_envs_test = {}

    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[400, 400, 400],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[400, 400, 400],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[400, 400, 400],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = SACReplayBuffer(env_spec=env.spec, max_size=int(1e6))
    sampler_args = {'agent': policy, 'max_path_length': 150}

    timesteps = 20000000
    batch_size = int(150 * env.num_tasks)
    num_evaluation_points = 500
    epochs = timesteps // batch_size
    epoch_cycles = epochs // num_evaluation_points
    epochs = epochs // epoch_cycles
    mtsac = MTSAC(env=env,
                  eval_env_dict=MT10_envs_test,
                  env_spec=env.spec,
                  policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  gradient_steps_per_itr=150,
                  epoch_cycles=epoch_cycles,
                  use_automatic_entropy_tuning=True,
                  replay_buffer=replay_buffer,
                  min_buffer_size=1500,
                  target_update_tau=5e-3,
                  discount=0.99,
                  buffer_batch_size=1280)
    tu.set_gpu_mode(True, _gpu)
    mtsac.to()

    runner.setup(algo=mtsac,
                 env=env,
                 sampler_cls=SimpleSampler,
                 sampler_args=sampler_args)

    runner.train(n_epochs=epochs, batch_size=batch_size)


s = np.random.randint(0, 1000)
mt10_sac_normalize_reward(seed=s)
