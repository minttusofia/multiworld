import gym
from gym.envs.registration import register
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)

_REGISTERED = False

print("\n\n\n\n\nin avi multiworld\n\n\n\n\n")

def register_goal_example_envs():

    register(id="Image48MetaworldDrawerOpenDense-v0",
            entry_point="multiworld.envs.mujoco.metaworld_image:MetaWorldDrawerOpenEnv",
            kwargs={'sparse_reward': False})

    register(id="Image48MetaworldDrawerOpenSparse-v0",
            entry_point="multiworld.envs.mujoco.metaworld_image:MetaWorldDrawerOpenEnv",
            kwargs={'sparse_reward': True})

    register(id="Image48MetaworldDrawerOpenSparse2D-v0",
            entry_point="multiworld.envs.mujoco.metaworld_image:MetaWorldDrawerOpenEnv",
            kwargs={'sparse_reward': True,
                    'two_dimensional': True})
    register(id="Image48MetaworldDrawerOpenSparse2D-v1",
            entry_point="multiworld.envs.mujoco.metaworld_image:MetaWorldDrawerOpenMirrorEnv",
            kwargs={'sparse_reward': True,
                    'two_dimensional': True})

    register(id="Image48MetaworldDrawerCloseSparse-v0",
            entry_point="multiworld.envs.mujoco.metaworld_image:MetaWorldDrawerOpenEnv",
            kwargs={'env_name': "DrawerClose", 'sparse_reward': True})

    register(id="AcrobotContinuous-v1",
            entry_point="multiworld.envs.mujoco.acrobot_continuous:AcrobotContinuousEnv")


    LOGGER.info("Registering goal example multiworld mujoco gym environments")

    """
    Door pull open tasks
    """
    #TODO Avi: A lot of code repetition here that we can get rid of

    register(
        id='BaseSawyerDoorHookEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_door_hook:SawyerDoorHookRandomInitEnv',
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        kwargs={
            'fix_goal': True,
            'fixed_goal': (0.02, 0.45, 0.15, 0.80),
            'indicator_threshold': (0.1, 0.05),
            'reward_type': 'angle_success',
            'hand_low': (-0.1, 0.30, 0.1),
            'hand_high': (0.05, 0.65, .40),
            'min_angle': 0.0,
            'max_angle': 0.83,
            'reset_free': False,
        }
        )

    register(
        id='BaseSawyerPushSidewaysEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachTargetObjectXYEnv',
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        kwargs={
            'fix_goal': True,
            'fixed_goal': (0.0, 0.6, 0.05, -0.15, 0.6),
            'indicator_threshold': 0.03,
            'reward_type': 'puck_success_positive',
            'puck_radius': 0.03,
            'reset_free': False,
            'xml_path': 'sawyer_xyz/sawyer_push_mug_to_coaster.xml',
            'hide_goal_markers': True,
            'puck_random_init': False,
        }
        )

    register(
        id='BaseSawyerPushForwardHandTouchPuckDistEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachTargetObjectXYEnv',
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        kwargs={
            'fix_goal': True,
            'fixed_goal': (0.0, 0.6, 0.05, 0.0, 0.75),
            'indicator_threshold': 0.03,
            'reward_type': 'hand_touch_puck_distance',
            'puck_radius': 0.03,
            'reset_free': False,
            'xml_path': 'sawyer_xyz/sawyer_push_mug_to_coaster.xml',
            'hide_goal_markers': True,
            'puck_random_init': True,
        }
        )

    register(
        id='BaseHumanLikeSawyerPushForwardHandTouchPuckDistEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachTargetObjectXYEnv',
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        kwargs={
            'fix_goal': True,
            'fixed_goal': (0.0, 0.6, 0.05, 0.0, 0.75),
            'indicator_threshold': 0.03,
            'reward_type': 'hand_touch_puck_distance',
            'puck_radius': 0.03,
            'reset_free': False,
            'xml_path': 'sawyer_xyz/sawyer_push_puck_to_square_human_like.xml',
            'hide_goal_markers': True,
            'puck_random_init': True,
        }
        )


    register(
        id='BaseRecoloredSawyerPushForwardHandTouchPuckDistEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachTargetObjectXYEnv',
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        kwargs={
            'fix_goal': True,
            'fixed_goal': (0.0, 0.6, 0.05, 0.0, 0.75),
            'indicator_threshold': 0.03,
            'reward_type': 'hand_touch_puck_distance',
            'puck_radius': 0.03,
            'reset_free': False,
            'xml_path': 'sawyer_xyz/sawyer_recolored_push_mug_to_coaster.xml',
            'hide_goal_markers': True,
            'puck_random_init': True,
        }
        )

    register(
        id='BaseSawyerPushForwardHandPuckDistEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachTargetObjectXYEnv',
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        kwargs={
            'fix_goal': True,
            'fixed_goal': (0.0, 0.6, 0.05, 0.0, 0.75),
            'indicator_threshold': 0.03,
            'reward_type': 'hand_and_puck_distance',
            'puck_radius': 0.03,
            'reset_free': False,
            'xml_path': 'sawyer_xyz/sawyer_push_mug_to_coaster.xml',
            'hide_goal_markers': True,
            'puck_random_init': True,
        }
        )

    register(
        id='BaseHumanLikeSawyerPushForwardPuckDistEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachTargetObjectXYEnv',
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        kwargs={
            'fix_goal': True,
            'fixed_goal': (0.0, 0.6, 0.05, 0.0, 0.75),
            'indicator_threshold': 0.03,
            'reward_type': 'puck_distance',
            'puck_radius': 0.03,
            'reset_free': False,
            'xml_path': 'sawyer_xyz/sawyer_push_puck_to_square_human_like.xml',
            'hide_goal_markers': True,
            'puck_random_init': True,
        }
        )

    register(
        id='BaseSawyerPushForwardPuckDistEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachTargetObjectXYEnv',
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        kwargs={
            'fix_goal': True,
            'fixed_goal': (0.0, 0.6, 0.05, 0.0, 0.75),
            'indicator_threshold': 0.03,
            'reward_type': 'puck_distance',
            'puck_radius': 0.03,
            'reset_free': False,
            'xml_path': 'sawyer_xyz/sawyer_push_mug_to_coaster.xml',
            'hide_goal_markers': True,
            'puck_random_init': True,
        }
        )


    register(
        id='BaseSawyerPushForwardHandPuckDistPuckSuccessEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachTargetObjectXYEnv',
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        kwargs={
            'fix_goal': True,
            'fixed_goal': (0.0, 0.6, 0.05, 0.0, 0.75),
            'indicator_threshold': 0.03,
            'reward_type': 'hand_and_puck_distance_and_puck_success',
            'puck_radius': 0.03,
            'reset_free': False,
            'xml_path': 'sawyer_xyz/sawyer_push_mug_to_coaster.xml',
            'hide_goal_markers': True,
            'puck_random_init': True,
        }
        )

    register(
        id='BaseHumanLikeSawyerPushForwardEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachTargetObjectXYEnv',
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        kwargs={
            'fix_goal': True,
            'fixed_goal': (0.0, 0.6, 0.05, 0.0, 0.75),
            'indicator_threshold': 0.03,
            'reward_type': 'puck_success_positive',
            'puck_radius': 0.03,
            'reset_free': False,
            'xml_path': 'sawyer_xyz/sawyer_push_puck_to_square_human_like.xml',
            'hide_goal_markers': True,
            'puck_random_init': True,
        }
        )

    register(
        id='BaseSawyerPushForwardEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachTargetObjectXYEnv',
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        kwargs={
            'fix_goal': True,
            'fixed_goal': (0.0, 0.6, 0.05, 0.0, 0.75),
            'indicator_threshold': 0.03,
            'reward_type': 'puck_success_positive',
            'puck_radius': 0.03,
            'reset_free': False,
            'xml_path': 'sawyer_xyz/sawyer_push_mug_to_coaster.xml',
            'hide_goal_markers': True,
            'puck_random_init': True,
        }
        )


    register(
        id='BaseRecoloredSawyerPushForwardEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachTargetObjectXYEnv',
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        kwargs={
            'fix_goal': True,
            'fixed_goal': (0.0, 0.6, 0.05, 0.0, 0.75),
            'indicator_threshold': 0.03,
            'reward_type': 'puck_success_positive',
            'puck_radius': 0.03,
            'reset_free': False,
            'xml_path': 'sawyer_xyz/sawyer_recolored_push_mug_to_coaster.xml',
            'hide_goal_markers': True,
            'puck_random_init': True,
        }
        )

    register(
        id='BaseSawyerPickAndPlaceEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place:SawyerPickAndPlaceEnvYZ',
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        kwargs={
                'hand_low': (0.0, 0.55, 0.01),
                'hand_high': (0.0, 0.65, 0.25),
                'action_scale': 0.02,
                'indicator_threshold': 0.03,
                'hide_goal_markers': True,
                'num_goals_presampled': 1,
                'reward_type': 'obj_success_positive',
                'p_obj_in_hand': .75,
                'fix_goal': True,
                'fixed_goal': (0.0, 0.6, 0.10, 0., 0.6, 0.20),
                'presampled_goals': {'state_desired_goal': np.asarray((0.0, 0.6, 0.10, 0., 0.6, 0.20)).reshape(1,6)},
                'obj_init_positions': ((0, 0.6, 0.02), (0, 0.62, 0.02), (0, 0.65, 0.02), (0, 0.57, 0.02), (0, 0.55, 0.02))
        }
        )

    register(
        id='BaseSawyerPickAndPlace3DEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place:SawyerPickAndPlaceEnv',
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        kwargs={
                'hand_low': (-0.05, 0.55, 0.01),
                'hand_high': (0.05, 0.65, 0.25),
                'obj_low': (-0.05, 0.55, 0.01),
                'obj_high': (0.05, 0.65, 0.25),
                'action_scale': 0.02,
                'indicator_threshold': 0.03,
                'hide_goal_markers': True,
                'num_goals_presampled': 1,
                'random_init': True,
                'reward_type': 'obj_success_positive',
                'p_obj_in_hand': .75,
                'fix_goal': True,
                'fixed_goal': (0.0, 0.6, 0.10, 0., 0.6, 0.20),
                'presampled_goals': {'state_desired_goal': np.asarray((0.0, 0.6, 0.10, 0., 0.6, 0.20)).reshape(1,6)},
            }
        )

    register(
        id='StateSawyerDoorPullHookEnv-v0',
        entry_point=create_state_sawyer_door_pull_hook_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        )

    register(
        id='Image48SawyerDoorPullHookEnv-v0',
        entry_point=create_image_48_sawyer_door_pull_hook_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        )

    register(
        id='StateSawyerPushForwardEnv-v0',
        entry_point=create_state_sawyer_push_forward_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        )

    register(
        id='Image48SawyerPushForwardHandPuckDistEnv-v0',
        entry_point=create_image_48_sawyer_push_forward_hand_puck_dist_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'karl'
        },
        )

    register(
        id='Image48HumanLikeSawyerPushForwardPuckDistEnv-v0',
        entry_point=create_image_48_human_like_sawyer_push_forward_puck_dist_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'karl'
        },
        )

    register(
        id='Image48SawyerPushForwardPuckDistEnv-v0',
        entry_point=create_image_48_sawyer_push_forward_puck_dist_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'karl'
        },
        )

    register(
        id='Image48SawyerPushForwardHandPuckDistPuckSuccessEnv-v0',
        entry_point=create_image_48_sawyer_push_forward_hand_puck_dist_puck_success_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'karl'
        },
        )

    register(
        id='Image48SawyerPushForwardHandTouchPuckDistEnv-v0',
        entry_point=create_image_48_sawyer_push_forward_hand_touch_puck_dist_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'karl'
        },
        )
    
    register(
        id='Image48RecoloredSawyerPushForwardHandTouchPuckDistEnv-v0',
        entry_point=create_image_48_recolored_sawyer_push_forward_hand_touch_puck_dist_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'karl'
        },
        )

    register(
        id='Image48HumanLikeSawyerPushForwardHandTouchPuckDistEnv-v0',
        entry_point=create_image_48_human_like_sawyer_push_forward_hand_touch_puck_dist_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'karl'
        },
        )

    register(
        id='Image48HumanLikeSawyerPushForwardEnv-v0',
        entry_point=create_image_48_human_like_sawyer_push_forward_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        )
    register(
        id='Image48HumanLikeSawyerPushForwardEnv-v1',
        entry_point=create_image_48_human_like_sawyer_push_forward_v1,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'minttu'
        },
        )
    register(
        id='Image48SawyerPushForwardEnv-v0',
        entry_point=create_image_48_sawyer_push_forward_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        )

    register(
        id='Image48RecoloredSawyerPushForwardEnv-v0',
        entry_point=create_image_48_recolored_sawyer_push_forward_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        )


    register(
        id='StateSawyerPushSidewaysEnv-v0',
        entry_point=create_state_sawyer_push_sideways_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        )

    register(
        id='Image48SawyerPushSidewaysEnv-v0',
        entry_point=create_image_48_sawyer_push_sideways_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        )

    register(
        id='StateSawyerPickAndPlaceEnv-v0',
        entry_point=create_state_sawyer_pick_and_place_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        )

    register(
        id='Image48SawyerPickAndPlaceEnv-v0',
        entry_point=create_image_48_sawyer_pick_and_place_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        )

    register(
        id='StateSawyerPickAndPlace3DEnv-v0',
        entry_point=create_state_sawyer_pick_and_place_3d_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        )

    register(
        id='Image48SawyerPickAndPlace3DEnv-v0',
        entry_point=create_image_48_sawyer_pick_and_place_3d_v0,
        tags={
            'git-commit-hash': '0de5200',
            'author': 'avi'
        },
        )

def create_state_sawyer_pick_and_place_3d_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    wrapped_env = gym.make('BaseSawyerPickAndPlace3DEnv-v0')
    return FlatGoalEnv(wrapped_env, obs_keys=['observation'])

def create_image_48_sawyer_pick_and_place_3d_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera_slanted_angle_zoomed
    wrapped_env = gym.make('BaseSawyerPickAndPlace3DEnv-v0')
    state_desired_goal = wrapped_env.fixed_goal
    goal_dim = len(state_desired_goal)
    imsize = 48
    image_env = ImageEnv(
        wrapped_env=wrapped_env,
        imsize=imsize,
        init_camera=sawyer_pick_and_place_camera_slanted_angle_zoomed,
        normalize=True,
        presampled_goals={'state_desired_goal': state_desired_goal.reshape(1,goal_dim),
                          'image_desired_goal': np.zeros((1, imsize*imsize*3))},
        )
    return FlatGoalEnv(image_env, obs_keys=['image_observation'])

def create_state_sawyer_pick_and_place_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    wrapped_env = gym.make('BaseSawyerPickAndPlaceEnv-v0')
    return FlatGoalEnv(wrapped_env, obs_keys=['observation'])

def create_image_48_sawyer_pick_and_place_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera_zoomed
    wrapped_env = gym.make('BaseSawyerPickAndPlaceEnv-v0')
    state_desired_goal = wrapped_env.fixed_goal
    goal_dim = len(state_desired_goal)
    imsize = 48
    image_env = ImageEnv(
        wrapped_env=wrapped_env,
        imsize=imsize,
        init_camera=sawyer_pick_and_place_camera_zoomed,
        normalize=True,
        presampled_goals={'state_desired_goal': state_desired_goal.reshape(1,goal_dim),
                          'image_desired_goal': np.zeros((1, imsize*imsize*3))},
        )
    return FlatGoalEnv(image_env, obs_keys=['image_observation'])

def create_state_sawyer_push_forward_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    wrapped_env = gym.make('BaseSawyerPushForwardEnv-v0')
    return FlatGoalEnv(wrapped_env, obs_keys=['observation'])

def create_image_48_human_like_sawyer_push_forward_hand_touch_puck_dist_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
    image_env = ImageEnv(
        wrapped_env=gym.make('BaseHumanLikeSawyerPushForwardHandTouchPuckDistEnv-v0'),
        imsize=48,
        init_camera=sawyer_pusher_camera_upright_v2,
        normalize=True,
        )
    return FlatGoalEnv(image_env, obs_keys=['image_observation'])

def create_image_48_recolored_sawyer_push_forward_hand_touch_puck_dist_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
    image_env = ImageEnv(
        wrapped_env=gym.make('BaseRecoloredSawyerPushForwardHandTouchPuckDistEnv-v0'),
        imsize=48,
        init_camera=sawyer_pusher_camera_upright_v2,
        normalize=True,
        )
    return FlatGoalEnv(image_env, obs_keys=['image_observation'])

def create_image_48_sawyer_push_forward_hand_touch_puck_dist_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
    image_env = ImageEnv(
        wrapped_env=gym.make('BaseSawyerPushForwardHandTouchPuckDistEnv-v0'),
        imsize=48,
        init_camera=sawyer_pusher_camera_upright_v2,
        normalize=True,
        )
    return FlatGoalEnv(image_env, obs_keys=['image_observation'])

def create_image_48_sawyer_push_forward_hand_puck_dist_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
    image_env = ImageEnv(
        wrapped_env=gym.make('BaseSawyerPushForwardHandPuckDistEnv-v0'),
        imsize=48,
        init_camera=sawyer_pusher_camera_upright_v2,
        normalize=True,
        )
    return FlatGoalEnv(image_env, obs_keys=['image_observation'])

def create_image_48_human_like_sawyer_push_forward_puck_dist_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
    image_env = ImageEnv(
        wrapped_env=gym.make('BaseHumanLikeSawyerPushForwardPuckDistEnv-v0'),
        imsize=48,
        init_camera=sawyer_pusher_camera_upright_v2,
        normalize=True,
        )
    return FlatGoalEnv(image_env, obs_keys=['image_observation'])

def create_image_48_sawyer_push_forward_puck_dist_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
    image_env = ImageEnv(
        wrapped_env=gym.make('BaseSawyerPushForwardPuckDistEnv-v0'),
        imsize=48,
        init_camera=sawyer_pusher_camera_upright_v2,
        normalize=True,
        )
    return FlatGoalEnv(image_env, obs_keys=['image_observation'])

def create_image_48_sawyer_push_forward_hand_puck_dist_puck_success_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
    image_env = ImageEnv(
        wrapped_env=gym.make('BaseSawyerPushForwardHandPuckDistPuckSuccessEnv-v0'),
        imsize=48,
        init_camera=sawyer_pusher_camera_upright_v2,
        normalize=True,
        )
    return FlatGoalEnv(image_env, obs_keys=['image_observation'])

def create_image_48_recolored_sawyer_push_forward_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
    image_env = ImageEnv(
        wrapped_env=gym.make('BaseRecoloredSawyerPushForwardEnv-v0'),
        imsize=48,
        init_camera=sawyer_pusher_camera_upright_v2,
        normalize=True,
        )
    return FlatGoalEnv(image_env, obs_keys=['image_observation'])

def create_image_48_human_like_sawyer_push_forward_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
    image_env = ImageEnv(
        wrapped_env=gym.make('BaseHumanLikeSawyerPushForwardEnv-v0'),
        imsize=48,
        init_camera=sawyer_pusher_camera_upright_v2,
        normalize=True,
        )
    return FlatGoalEnv(image_env, obs_keys=['image_observation'])

def create_image_48_human_like_sawyer_push_forward_v1():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    from multiworld.core.image_env import ImageEnv
    from multiworld.core.flip_env import FlipEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
    image_env = ImageEnv(
        wrapped_env=gym.make('BaseHumanLikeSawyerPushForwardEnv-v0'),
        imsize=48,
        init_camera=sawyer_pusher_camera_upright_v2,
        normalize=True,
        )
    goal_env = FlatGoalEnv(image_env, obs_keys=['image_observation'])
    return FlipEnv(goal_env)

def create_image_48_sawyer_push_forward_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
    image_env = ImageEnv(
        wrapped_env=gym.make('BaseSawyerPushForwardEnv-v0'),
        imsize=48,
        init_camera=sawyer_pusher_camera_upright_v2,
        normalize=True,
        )
    return FlatGoalEnv(image_env, obs_keys=['image_observation'])

def create_state_sawyer_push_sideways_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    wrapped_env = gym.make('BaseSawyerPushSidewaysEnv-v0')
    return FlatGoalEnv(wrapped_env, obs_keys=['observation'])

def create_image_48_sawyer_push_sideways_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
    
    image_env = ImageEnv(
        wrapped_env = gym.make('BaseSawyerPushSidewaysEnv-v0'),
        imsize=48,
        init_camera=sawyer_pusher_camera_upright_v2,
        normalize=True,
        )
    return FlatGoalEnv(image_env, obs_keys=['image_observation'])

def create_state_sawyer_door_pull_hook_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    wrapped_env = gym.make('BaseSawyerDoorHookEnv-v0')
    return FlatGoalEnv(wrapped_env, obs_keys=['observation'])

def create_image_48_sawyer_door_pull_hook_v0():
    from multiworld.core.flat_goal_env import FlatGoalEnv
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v0
    import numpy as np

    wrapped_env = gym.make('BaseSawyerDoorHookEnv-v0')
    imsize=48
    imsize_flat=imsize*imsize*3
    image_env = ImageEnv(
        wrapped_env=wrapped_env,
        imsize=imsize,
        init_camera=sawyer_door_env_camera_v0,
        normalize=True,
        presampled_goals={
        'state_desired_goal': 
        np.expand_dims(wrapped_env.fixed_goal, axis=0),
        'image_desired_goal':
        np.zeros((1, imsize_flat))},
        non_presampled_goal_img_is_garbage=True,
        )
    return FlatGoalEnv(image_env, obs_keys=['image_observation'])

def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    LOGGER.info("Registering multiworld mujoco gym environments")
    from multiworld.envs.mujoco.cameras import (
        sawyer_init_camera_zoomed_in
    )
    """
    Reaching tasks
    """

    register(
        id='SawyerReachXYEnv-v1',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_reach:SawyerReachXYEnv',
        tags={
            'git-commit-hash': '2d95c75',
            'author': 'murtaza'
        },
        kwargs={
            'hide_goal_markers': True,
            'norm_order': 2,
        },
    )

    register(
        id='SawyerReachXYZEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_reach:SawyerReachXYZEnv',
        tags={
            'git-commit-hash': '7b3113b',
            'author': 'vitchyr'
        },
        kwargs={
            'hide_goal_markers': False,
            'norm_order': 2,
        },
    )

    register(
        id='SawyerReachXYZEnv-v1',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_reach:SawyerReachXYZEnv',
        tags={
            'git-commit-hash': 'bea5de',
            'author': 'murtaza'
        },
        kwargs={
            'hide_goal_markers': True,
            'norm_order': 2,
        },
    )

    register(
        id='Image48SawyerReachXYEnv-v1',
        entry_point=create_image_48_sawyer_reach_xy_env_v1,
        tags={
            'git-commit-hash': '2d95c75',
            'author': 'murtaza'
        },
    )
    register(
        id='Image84SawyerReachXYEnv-v1',
        entry_point=create_image_84_sawyer_reach_xy_env_v1,
        tags={
            'git-commit-hash': '2d95c75',
            'author': 'murtaza'
        },
    )


    """
    Pushing Tasks, XY
    """

    register(
        id='SawyerPushAndReachEnvEasy-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'ddd73dc',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.15, 0.4, 0.02, -.1, .45),
            goal_high=(0.15, 0.7, 0.02, .1, .65),
            puck_low=(-.1, .45),
            puck_high=(.1, .65),
            hand_low=(-0.15, 0.4, 0.02),
            hand_high=(0.15, .7, 0.02),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck.xml',
            reward_type='state_distance',
            reset_free=False,
            clamp_puck_on_step=True,
        )
    )

    register(
        id='SawyerPushAndReachEnvMedium-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'ddd73dc',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.2, 0.35, 0.02, -.15, .4),
            goal_high=(0.2, 0.75, 0.02, .15, .7),
            puck_low=(-.15, .4),
            puck_high=(.15, .7),
            hand_low=(-0.2, 0.35, 0.05),
            hand_high=(0.2, .75, 0.3),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck.xml',
            reward_type='state_distance',
            reset_free=False,
            clamp_puck_on_step=True,
        )
    )

    register(
        id='SawyerPushAndReachEnvHard-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'ddd73dc',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.25, 0.3, 0.02, -.2, .35),
            goal_high=(0.25, 0.8, 0.02, .2, .75),
            puck_low=(-.2, .35),
            puck_high=(.2, .75),
            hand_low=(-0.25, 0.3, 0.02),
            hand_high=(0.25, .8, 0.02),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck.xml',
            reward_type='state_distance',
            reset_free=False,
            clamp_puck_on_step=True,
        )
    )

    """
    Pushing tasks, XY, Arena
    """
    register(
        id='SawyerPushAndReachArenaEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'dea1627',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck_arena.xml',
            reward_type='state_distance',
            reset_free=False,
        )
    )

    register(
        id='SawyerPushAndReachArenaResetFreeEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'dea1627',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck_arena.xml',
            reward_type='state_distance',
            reset_free=True,
        )
    )

    register(
        id='SawyerPushAndReachSmallArenaEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '7256aaf',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.15, 0.4, 0.02, -.1, .5),
            goal_high=(0.15, 0.75, 0.02, .1, .7),
            puck_low=(-.3, .25),
            puck_high=(.3, .9),
            hand_low=(-0.15, 0.4, 0.05),
            hand_high=(0.15, .75, 0.3),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck_small_arena.xml',
            reward_type='state_distance',
            reset_free=False,
            clamp_puck_on_step=False,
        )
    )

    register(
        id='SawyerPushAndReachSmallArenaResetFreeEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '7256aaf',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.15, 0.4, 0.02, -.1, .5),
            goal_high=(0.15, 0.75, 0.02, .1, .7),
            puck_low=(-.3, .25),
            puck_high=(.3, .9),
            hand_low=(-0.15, 0.4, 0.05),
            hand_high=(0.15, .75, 0.3),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck_small_arena.xml',
            reward_type='state_distance',
            reset_free=True,
            clamp_puck_on_step=False,
        )
    )

    """
    NIPS submission pusher environment
    """
    register(
        id='SawyerPushNIPS-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_nips:SawyerPushAndReachXYEasyEnv',
        tags={
            'git-commit-hash': 'bede25d',
            'author': 'ashvin',
        },
        kwargs=dict(
            hide_goal=True,
            reward_info=dict(
                type="state_distance",
            ),
        )

    )

    register(
        id='SawyerPushNIPSHarder-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_nips:SawyerPushAndReachXYHarderEnv',
        tags={
            'git-commit-hash': 'b5cac93',
            'author': 'murtaza',
        },
        kwargs=dict(
            hide_goal=True,
            reward_info=dict(
                type="state_distance",
            ),
        )

    )

    """
    Door Hook Env
    """

    register(
        id='SawyerDoorHookEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_door_hook:SawyerDoorHookEnv',
        tags={
            'git-commit-hash': '15b48d5',
            'author': 'murtaza',
        },
        kwargs = dict(
            goal_low=(-0.1, 0.45, 0.1, 0),
            goal_high=(0.05, 0.65, .25, .83),
            hand_low=(-0.1, 0.45, 0.1),
            hand_high=(0.05, 0.65, .25),
            max_angle=.83,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
            reward_type='angle_diff_and_hand_distance',
            reset_free=False,
        )
    )

    register(
        id='SawyerDoorHookResetFreeEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_door_hook:SawyerDoorHookEnv',
        tags={
            'git-commit-hash': '15b48d5',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.1, 0.45, 0.1, 0),
            goal_high=(0.05, 0.65, .25, .83),
            hand_low=(-0.1, 0.45, 0.1),
            hand_high=(0.05, 0.65, .25),
            max_angle=.83,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
            reward_type='angle_diff_and_hand_distance',
            reset_free=True,
        )
    )

    """
    Pick and Place
    """
    register(
        id='SawyerPickupEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnv',
        tags={
            'git-commit-hash': '30f23f7',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.2),
            action_scale=0.02,
            hide_goal_markers=True,
            num_goals_presampled=1000,
        )

    )
    register(
        id='SawyerPickupResetFreeEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnv',
        tags={
            'git-commit-hash': '30f23f7',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.2),
            action_scale=0.02,
            reset_free=True,
            hide_goal_markers=True,
            num_goals_presampled=1000,
        )

    )
    register(
        id='SawyerPickupEnvYZ-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnvYZ',
        tags={
            'git-commit-hash': '30f23f7',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.2),
            action_scale=0.02,
            hide_goal_markers=True,
            num_goals_presampled=1000,
        )
    )


    register(
        id='SawyerPickupTallEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnv',
        tags={
            'git-commit-hash': '30f23f7',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.3),
            action_scale=0.02,
            hide_goal_markers=True,
            num_goals_presampled=1000,
        )

    )
    register(
        id='SawyerPickupWideEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnv',
        tags={
            'git-commit-hash': '30f23f7',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.2, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.2),
            action_scale=0.02,
            hide_goal_markers=True,
            num_goals_presampled=1000,
        )

    )

    register(
        id='SawyerPickupWideResetFreeEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnv',
        tags={
            'git-commit-hash': '30f23f7',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.2, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.2),
            action_scale=0.02,
            hide_goal_markers=True,
            reset_free=True,
            num_goals_presampled=1000,
        )

    )


    register(
        id='SawyerPickupTallWideEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnv',
        tags={
            'git-commit-hash': '30f23f7',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.2, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.3),
            action_scale=0.02,
            hide_goal_markers=True,
            num_goals_presampled=1000,
        )

    )

    """
    ICML Envs
    """
    register(
        id='SawyerPickupEnvYZEasy-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnvYZ',
        tags={
            'git-commit-hash': '8bfd74b40f983e15026981344323b8e9539b4b21',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.13),
            action_scale=0.02,
            hide_goal_markers=True,
            num_goals_presampled=1000,

            p_obj_in_hand=.75,
        )
    )
    # This env is used for the image pickup version. We don't need state goals,
    # as the image env already generates image goals + state goals.
    register(
        id='SawyerPickupEnvYZEasyFewGoals-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnvYZ',
        tags={
            'git-commit-hash': '8bfd74b40f983e15026981344323b8e9539b4b21',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.13),
            action_scale=0.02,
            hide_goal_markers=True,
            num_goals_presampled=1,

            p_obj_in_hand=.75,
        )
    )

    register(
        id='SawyerPickupEnvYZEasyImage48-v0',
        entry_point=create_image_48_sawyer_pickup_easy_v0,
        tags={
            'git-commit-hash': '8bfd74b40f983e15026981344323b8e9539b4b21',
            'author': 'steven'
        },
    )
    register(
        id='SawyerDoorHookResetFreeEnvImage48-v1',
        entry_point=create_image_48_sawyer_door_hook_reset_free_v1,
        tags={
            'git-commit-hash': '8bfd74b40f983e15026981344323b8e9539b4b21',
            'author': 'steven'
        },
    )
    register(
        id='SawyerPushNIPSEasy-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_nips:SawyerPushAndReachXYEasyEnv',
        tags={
            'git-commit-hash': 'b8d77fef5f3ebe4c1c9c3874a5e3faaab457a350',
            'author': 'steven',
        },
        kwargs=dict(
            force_puck_in_goal_space=False,
            mocap_low=(-0.1, 0.55, 0.0),
            mocap_high=(0.1, 0.65, 0.5),
            hand_goal_low=(-0.1, 0.55),
            hand_goal_high=(0.1, 0.65),
            puck_goal_low=(-0.15, 0.5),
            puck_goal_high=(0.15, 0.7),

            hide_goal=True,
            reward_info=dict(
                type="state_distance",
            ),
        )
    )
    register(
        id='SawyerPushNIPSEasyImage48-v0',
        entry_point='multiworld.core.image_env:ImageEnv',
        tags={
            'git-commit-hash': 'b8d77fef5f3ebe4c1c9c3874a5e3faaab457a350',
            'author': 'steven',
        },
        kwargs=dict(
            wrapped_env=gym.make('SawyerPushNIPSEasy-v0'),
            imsize=48,
            init_camera=sawyer_init_camera_zoomed_in,
            transpose=True,
            normalize=True,
        )
    )
    register(
        id='SawyerDoorHookResetFreeEnv-v1',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_door_hook:SawyerDoorHookEnv',
        tags={
            'git-commit-hash': 'b8d77fef5f3ebe4c1c9c3874a5e3faaab457a350',
            'author': 'steven',
        },
        kwargs=dict(
            goal_low=(-0.1, 0.45, 0.1, 0),
            goal_high=(0.05, 0.65, .25, .83),
            hand_low=(-0.1, 0.45, 0.1),
            hand_high=(0.05, 0.65, .25),
            max_angle=.83,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
            reward_type='angle_diff_and_hand_distance',
            reset_free=True,
        )
    )

    register(
        id='SawyerReachXYZEnv-v2',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_reach:SawyerReachXYZEnv',
        tags={
            'git-commit-hash': 'b8d77fef5f3ebe4c1c9c3874a5e3faaab457a350',
            'author': 'steven'
        },
        kwargs={
            'hide_goal_markers': True,
            'norm_order': 2,
        },
    )


def create_image_48_sawyer_reach_xy_env_v1():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_xyz_reacher_camera_v0

    wrapped_env = gym.make('SawyerReachXYEnv-v1')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_xyz_reacher_camera_v0,
        transpose=True,
        normalize=True,
    )


def create_image_84_sawyer_reach_xy_env_v1():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_xyz_reacher_camera_v0

    wrapped_env = gym.make('SawyerReachXYEnv-v1')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=sawyer_xyz_reacher_camera_v0,
        transpose=True,
        normalize=True,
    )

def create_image_48_sawyer_push_and_reach_arena_env_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

    wrapped_env = gym.make('SawyerPushAndReachArenaEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_pusher_camera_upright_v2,
        transpose=True,
        normalize=True,
    )

def create_image_48_sawyer_push_and_reach_arena_env_reset_free_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

    wrapped_env = gym.make('SawyerPushAndReachArenaResetFreeEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_pusher_camera_upright_v2,
        transpose=True,
        normalize=True,
    )

def create_image_48_sawyer_door_hook_reset_free_v1():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v0
    import os.path
    import numpy as np
    goal_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'goals/door_goals.npy'
    )
    goals = np.load(goal_path).item()
    return ImageEnv(
        wrapped_env=gym.make('SawyerDoorHookResetFreeEnv-v1'),
        imsize=48,
        init_camera=sawyer_door_env_camera_v0,
        transpose=True,
        normalize=True,
        presampled_goals=goals,
    )

def create_image_48_sawyer_pickup_easy_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
    import os.path
    import numpy as np
    goal_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'goals/pickup_goals.npy'
    )
    goals = np.load(goal_path).item()
    return ImageEnv(
        wrapped_env=gym.make('SawyerPickupEnvYZEasyFewGoals-v0'),
        imsize=48,
        init_camera=sawyer_pick_and_place_camera,
        transpose=True,
        normalize=True,
        presampled_goals=goals,
    )


register_custom_envs()
