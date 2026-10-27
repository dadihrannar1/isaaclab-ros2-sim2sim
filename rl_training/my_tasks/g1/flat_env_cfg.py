from isaaclab.envs.mdp import UniformVelocityCommandCfg
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

        # Commands: (vx, vy, wz) each sampled uniformly from its range. Values below = stock G1 flat,
        # except lin_vel_x which also trains walking backwards.
        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (-1.0, 1.0)
        cmd.ranges.lin_vel_y = (-0.5, 0.5)
        cmd.ranges.ang_vel_z = (-1.0, 1.0)
        cmd.heading_command = False  # True: wz computed from a random heading target; False: wz sampled directly
        cmd.resampling_time_range = (10.0, 10.0)  # seconds between new commands
        cmd.rel_standing_envs = 0.1  # fraction of commands that are "stand still" (stock: 0.02)

        # Old approach: penalize feet in the air while standing. Hurt the walking gait, kept for reference.
        # self.rewards.stationary_feet_airtime = RewTerm(
        #     func=mdp.stationary_feet_airtime,
        #     weight=-0.5,
        #     params={
        #         "command_name": "base_velocity",
        #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        #     },
        # )

        # Stand still: pull leg joints back to the default pose when the command is ~zero
        self.rewards.stand_still = RewTerm(
            func=mdp.stand_still_joint_deviation,
            weight=-0.3,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]),
            },
        )

        # weaker penalty (stock: -0.1)
        # self.rewards.joint_deviation_hip.weight = 0.0

        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25  # stock 0.5; standing under 0.3 m/s now earns ~24% instead of ~70%



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


# @configclass
# class MyG1FlatEnvCfg_PLAY_SIDESTEP(MyG1FlatEnvCfg_PLAY):
#     """Play with a fixed pure-sideways command (vy = 0.5 m/s) to test side-stepping."""

#     def __post_init__(self) -> None:
#         super().__post_init__()

#         # plain uniform command with fixed ranges -> every robot always gets exactly (0, 0.5, 0)
#         self.commands.base_velocity = UniformVelocityCommandCfg(
#             asset_name="robot",
#             resampling_time_range=(10.0, 10.0),
#             rel_standing_envs=0.0,
#             heading_command=False,
#             debug_vis=True,
#             ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.5, 0.5), ang_vel_z=(0.0, 0.0)),
#         )
