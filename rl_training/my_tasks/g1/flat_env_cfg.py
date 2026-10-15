from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

from my_tasks import mdp


@configclass
class MyG1FlatEnvCfg(G1FlatEnvCfg):
    """Stock G1 flat task with our own tweaks. Only list what differs from the parent."""

    def __post_init__(self):
        # post init of parent (sets up the whole stock G1 flat task)
        super().__post_init__()

        # Example tweaks -- change these to experiment
        # Rewards
        #self.rewards.feet_air_time.weight = 1.0
        #self.rewards.flat_orientation_l2.weight = -2.0
        # Commands: let it walk a bit faster forward
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)

        self.rewards.standing_still_feet_air = RewTerm(
            func=mdp.standing_still_feet_air,
            weight=-0.75,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            },
        )


@configclass
class MyG1FlatEnvCfg_PLAY(MyG1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
