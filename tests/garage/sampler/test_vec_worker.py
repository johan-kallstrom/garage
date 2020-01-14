from unittest.mock import Mock

from garage.envs.grid_world_env import GridWorldEnv
from garage.np.policies import ScriptedPolicy
from garage.sampler import VecWorker
from garage.tf.envs import TfEnv


class TestVecWorker:

    def setup_method(self):
        self.seed = 100
        self.max_path_length = 16
        self.env = TfEnv(GridWorldEnv(desc='4x4'))
        self.policy = ScriptedPolicy(
            scripted_actions=[2, 2, 1, 0, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1])
        self.algo = Mock(env_spec=self.env.spec,
                         policy=self.policy,
                         max_path_length=self.max_path_length)

    def teardown_method(self):
        self.env.close()

    def test_rollout(self):
        worker = VecWorker(seed=self.seed,
                           max_path_length=self.max_path_length,
                           worker_number=0)
        worker.update_agent(self.policy)
        worker.update_env(self.env)
        traj = worker.rollout()
        assert len(traj.lengths) == 8
        traj2 = worker.rollout()
        assert len(traj2.lengths) == 8
        worker.shutdown()
