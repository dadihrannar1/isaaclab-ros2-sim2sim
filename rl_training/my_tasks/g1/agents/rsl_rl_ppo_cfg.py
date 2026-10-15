from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class MyG1FlatPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # logs go to logs/rsl_rl/my_g1_flat/ so they don't mix with stock runs
        self.experiment_name = "my_g1_flat"
