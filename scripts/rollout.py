from robosuite_benchmark.util.rlkit_utils import simulate_policy
from robosuite_benchmark.util.arguments import add_rollout_args, parser
import robosuite as suite
from robosuite.wrappers import GymWrapper
# from robosuite.controllers import ALL_CONTROLLERS, load_controller_config
import numpy as np
import torch
import imageio
import os
import json

from signal import signal, SIGINT
from sys import exit

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

# Add and parse arguments
add_rollout_args()
args = parser.parse_args()

# Define callbacks
video_writer = None


def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Closing video writer and exiting gracefully')
    video_writer.close()
    exit(0)


# Tell Python to run the handler() function when SIGINT is recieved
signal(SIGINT, handler)

import textwrap

class ObsEnvWrapper:
    def __init__(self, env, offset=0):
        self.env = env
        self.offset = offset

    def filter_observation(self, obs):
        return obs
        obs_shards = [obs[:7], obs[self.offset-1:self.offset], obs[7:self.offset-1],obs[self.offset+7:35], obs[39:]]
        # obs_shards = [obs[:self.offset],obs[self.offset+7:35], obs[39:]]
        
        o = torch.concat(obs_shards) if isinstance(obs, torch.Tensor) else np.concatenate(obs_shards)
        return o
    def reset(self):
        obs = self.env.reset()[0]
        obs = self.filter_observation(obs)
        return obs
    
    def step(self, *args, **kwargs):
        r = self.env.step(*args, **kwargs)
        # Filter out the observations
        obs = self.filter_observation(r[0])
        
        return obs, r[1], r[2], r[3]

    def _to_string(self):
        """
        Subclasses should override this method to print out info about the 
        wrapper (such as arguments passed to it).
        """
        return ''
    
    def __repr__(self):
        """Pretty print environment."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        if self._to_string() != '':
            msg += textwrap.indent("\n" + self._to_string(), indent)
        msg += textwrap.indent("\nenv={}".format(self.env), indent)
        msg = header + '(' + msg + '\n)'
        return msg

    # this method is a fallback option on any methods the original env might support
    def __getattr__(self, attr):
        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        orig_attr = getattr(self.env, attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if id(result) == id(self.env):
                    return self
                return result

            return hooked
        else:
            return orig_attr
        
if __name__ == "__main__":
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Get path to saved model
    kwargs_fpath = os.path.join(args.load_dir, "variant.json")
    try:
        with open(kwargs_fpath) as f:
            kwargs = json.load(f)
    except FileNotFoundError:
        print("Error opening default controller filepath at: {}. "
              "Please check filepath and try again.".format(kwargs_fpath))

    # Grab / modify env args
    env_args = kwargs["eval_environment_kwargs"]
    if args.horizon is not None:
        env_args["horizon"] = args.horizon
    env_args["render_camera"] = args.camera
    env_args["hard_reset"] = True
    env_args["ignore_done"] = True

    # Specify camera name if we're recording a video
    if args.record_video:
        env_args["camera_names"] = args.camera
        env_args["camera_heights"] = 512
        env_args["camera_widths"] = 512

    # Setup video recorder if necesssary
    if args.record_video:
        # Grab name of this rollout combo
        video_name = "{}-{}-{}".format(
            env_args["env_name"], "".join(env_args["robots"]), env_args["controller"]).replace("_", "-")
        # Calculate appropriate fps
        fps = int(env_args["control_freq"])
        # Define video writer
        video_writer = imageio.get_writer("{}.mp4".format(video_name), fps=fps)

    # Pop the controller
    controller = env_args.pop("controller")
    # if controller in ALL_CONTROLLERS:
    #     controller_config = load_controller_config(default_controller=controller)
    # else:
    #     controller_config = load_controller_config(custom_fpath=controller)
    # print(controller_config)
    # breakpoint()
    # controller_config = {'type': 'BASIC', 'body_parts': {'right': {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], 'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'damping_ratio': 1, 'impedance_mode': 'fixed', 'kp_limits': [0, 300], 'damping_ratio_limits': [0, 10], 'position_limits': None, 'orientation_limits': None, 'uncouple_pos_ori': True, 'input_type': 'delta', 'input_ref_frame': 'base', 'interpolation': None, 'ramp_ratio': 0.2, 'gripper': {'type': 'GRIP'}}, 'left': {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], 'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'damping_ratio': 1, 'impedance_mode': 'fixed', 'kp_limits': [0, 300], 'damping_ratio_limits': [0, 10], 'position_limits': None, 'orientation_limits': None, 'uncouple_pos_ori': True, 'input_type': 'delta', 'input_ref_frame': 'base', 'interpolation': None, 'ramp_ratio': 0.2, 'gripper': {'type': 'GRIP'}}, 'torso': {'type': 'JOINT_POSITION', 'input_max': 1, 'input_min': -1, 'output_max': 0.5, 'output_min': -0.5, 'kd': 200, 'kv': 200, 'kp': 1000, 'velocity_limits': [-1, 1], 'kp_limits': [0, 1000], 'interpolation': None, 'ramp_ratio': 0.2}, 'head': {'type': 'JOINT_POSITION', 'input_max': 1, 'input_min': -1, 'output_max': 0.5, 'output_min': -0.5, 'kd': 200, 'kv': 200, 'kp': 1000, 'velocity_limits': [-1, 1], 'kp_limits': [0, 1000], 'interpolation': None, 'ramp_ratio': 0.2}, 'base': {'type': 'JOINT_VELOCITY', 'interpolation': 'null'}, 'legs': {'type': 'JOINT_POSITION', 'input_max': 1, 'input_min': -1, 'output_max': 0.5, 'output_min': -0.5, 'kd': 200, 'kv': 200, 'kp': 1000, 'velocity_limits': [-1, 1], 'kp_limits': [0, 1000], 'interpolation': None, 'ramp_ratio': 0.2}}}

    #{'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], 'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'damping_ratio': 1, 'impedance_mode': 'fixed', 'kp_limits': [0, 300], 'damping_ratio_limits': [0, 10], 'position_limits': None, 'orientation_limits': None, 'uncouple_pos_ori': True, 'control_delta': True, 'interpolation': None, 'ramp_ratio': 0.2}
    # Create env
    env_suite = suite.make(**env_args,
                        #    controller_configs=controller_config,
                           has_renderer=not args.record_video,
                           has_offscreen_renderer=args.record_video,
                           use_object_obs=True,
                           use_camera_obs=args.record_video,
                           reward_shaping=True
                           )
    
    # Make sure we only pass in the proprio and object obs (no images)
    keys = ["object-state"]
    for idx in range(len(env_suite.robots)):
        keys.append(f"robot{idx}_proprio-state")
    # keys = ["object-state","robot0_joint_pos_cos",
	# 	"robot0_joint_pos_sin", "robot0_joint_vel",
	# 	"robot0_eef_pos", "robot0_eef_quat",
	# 	"robot0_gripper_qpos", "robot0_gripper_qvel"]
    
    # Wrap environment so it's compatible with Gym API
    env = GymWrapper(env_suite, keys=keys)
    if suite.__version__ >= "1.5.0":
        obs = env_suite.reset()
        env = ObsEnvWrapper(env, offset=obs["object-state"].shape[0])
    # Run rollout
    simulate_policy(
        env=env,
        model_path=os.path.join(args.load_dir, "params.pkl"),
        horizon=env_args["horizon"],
        render=not args.record_video,
        video_writer=video_writer,
        num_episodes=args.num_episodes,
        printout=True,
        use_gpu=args.gpu,
    )
