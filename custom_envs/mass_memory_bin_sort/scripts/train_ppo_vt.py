# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from collections import defaultdict
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
from gymnasium.vector import AsyncVectorEnv
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import custom_envs.sort_ycb_into_bins  # noqa: F401
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "PPO"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    resume_latest: bool = False
    """resume from the latest checkpoint in runs/<exp-name> (or auto run name if exp-name omitted)"""
    render_mode: str = "rgb_array"
    """the environment rendering mode"""
    render_width: int = 1024
    """width of rendered videos in pixels"""
    render_height: int = 1024
    """height of rendered videos in pixels"""
    render_shader: str = "default"
    """shader pack for rendered videos (e.g. minimal, default, rt-fast if supported)"""
    enable_shadow: bool = True
    """enable scene shadows when rendering videos"""
    video_fps: int = 30
    """fps used for saved videos"""
    video_quality: Optional[int] = 9
    """imageio quality in [0,10]; higher means better quality and larger files"""
    video_codec: str = "libx264"
    """ffmpeg video codec for video writing"""
    video_crf: Optional[int] = 17
    """ffmpeg CRF (lower is higher quality). Ignored if None"""
    video_preset: str = "slow"
    """ffmpeg preset for encoding efficiency/quality tradeoff"""
    video_bitrate: Optional[str] = None
    """optional ffmpeg bitrate override (e.g. 8M). If set, CRF/quality are less impactful"""
    video_pixel_format: str = "yuv420p"
    """pixel format used for encoded videos"""
    include_depth: bool = False
    """whether to include depth observations in addition to RGB"""
    sim_backend: str = "physx_cpu"
    """simulation backend: use physx_cpu on macOS"""

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    include_state: bool = True
    """whether to include state information in observations"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel environments"""
    num_eval_envs: int = 1
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.2
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    eval_during_train: bool = True
    """run periodic evaluation during training; disable on macOS if Vulkan device-lost occurs"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def _squeeze_leading_singleton(obs):
    if isinstance(obs, dict):
        return {k: _squeeze_leading_singleton(v) for k, v in obs.items()}
    if isinstance(obs, (np.ndarray, torch.Tensor)) and obs.ndim >= 1 and obs.shape[0] == 1:
        return obs[0]
    if isinstance(obs, (np.ndarray, torch.Tensor)) and obs.ndim >= 2 and obs.shape[1] == 1:
        return obs[:, 0]
    return obs


def _singleton_to_scalar(x):
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError(f"Expected singleton tensor, got shape {tuple(x.shape)}")
        return x.detach().cpu().reshape(()).item()
    if isinstance(x, np.ndarray):
        if x.size != 1:
            raise ValueError(f"Expected singleton array, got shape {x.shape}")
        return x.reshape(()).item()
    if isinstance(x, (list, tuple)):
        if len(x) != 1:
            raise ValueError(f"Expected singleton sequence, got length {len(x)}")
        return _singleton_to_scalar(x[0])
    return x


def _to_torch_tree(x, device):
    if isinstance(x, dict):
        return {k: _to_torch_tree(v, device) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, np.ndarray):
        if x.dtype == np.object_:
            return x
        return torch.from_numpy(x).to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_torch_tree(v, device) for v in x)
    return x


def _to_numpy_tree(x):
    if isinstance(x, dict):
        return {k: _to_numpy_tree(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return type(x)(_to_numpy_tree(v) for v in x)
    return x


def _squeezed_space(space):
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({k: _squeezed_space(v) for k, v in space.spaces.items()})
    if isinstance(space, gym.spaces.Box):
        if len(space.shape) >= 1 and space.shape[0] == 1:
            return gym.spaces.Box(low=space.low[0], high=space.high[0], shape=space.shape[1:], dtype=space.dtype)
        return space
    return space


class AsyncCPUVectorAdapter:
    """Adapter that makes AsyncVectorEnv behave similarly to ManiSkillVectorEnv for this script."""

    def __init__(self, env, device):
        self._env = env
        self.device = device
        self.action_space = env.action_space
        self.single_action_space = env.single_action_space
        self.single_observation_space = _squeezed_space(env.single_observation_space)
        self.unwrapped = self

    def reset(self, seed=None):
        obs, info = self._env.reset(seed=seed)
        obs = _squeeze_leading_singleton(obs)
        return _to_torch_tree(obs, self.device), info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        obs, reward, terminations, truncations, infos = self._env.step(action)
        obs = _to_torch_tree(_squeeze_leading_singleton(obs), self.device)
        reward = _to_torch_tree(_squeeze_leading_singleton(reward), self.device).to(torch.float32)
        terminations = _to_torch_tree(_squeeze_leading_singleton(terminations), self.device).to(torch.bool)
        truncations = _to_torch_tree(_squeeze_leading_singleton(truncations), self.device).to(torch.bool)
        # Async vector infos are not shape-compatible with existing final_info logging logic.
        # We keep rollout training stable and lightweight by skipping info-dependent logging here.
        return obs, reward, terminations, truncations, {}

    def close(self):
        self._env.close()


class AsyncCPUWorkerEnvAdapter(gym.Wrapper):
    """Adapts ManiSkill singleton env outputs to AsyncVectorEnv's scalar worker API."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = _squeezed_space(env.observation_space)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return _to_numpy_tree(_squeeze_leading_singleton(obs)), info

    def step(self, action):
        obs, reward, terminations, truncations, infos = self.env.step(action)
        obs = _to_numpy_tree(_squeeze_leading_singleton(obs))
        reward = float(_singleton_to_scalar(reward))
        terminations = bool(_singleton_to_scalar(terminations))
        truncations = bool(_singleton_to_scalar(truncations))
        return obs, reward, terminations, truncations, infos

class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)

class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]
        if "depth" in sample_obs:
            in_channels += sample_obs["depth"].shape[-1]
        image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])


        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            rgb = sample_obs["rgb"].float() / 255.0
            if "depth" in sample_obs:
                depth = sample_obs["depth"].float() / 1000.0
                rgb = torch.cat([rgb, depth], dim=-1)
            n_flatten = cnn(rgb.permute(0, 3, 1, 2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                rgb = obs.float() / 255.0
                if "depth" in observations:
                    # Depth is stored in millimeters for ManiSkill camera outputs.
                    depth = observations["depth"].float() / 1000.0
                    rgb = torch.cat([rgb, depth], dim=-1)
                obs = rgb.permute(0, 3, 1, 2)
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)

class Agent(nn.Module):
    def __init__(self, envs, sample_obs):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        # latent_size = np.array(envs.unwrapped.single_observation_space.shape).prod()
        latent_size = self.feature_net.out_features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, np.prod(envs.unwrapped.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.unwrapped.single_action_space.shape)) * -0.5)
    def get_features(self, x):
        return self.feature_net(x)
    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()


class ConfiguredRecordEpisode(RecordEpisode):
    """RecordEpisode variant with safer frame handling and configurable ffmpeg quality settings."""

    def __init__(
        self,
        *args,
        video_quality: Optional[int] = 9,
        video_codec: str = "libx264",
        video_crf: Optional[int] = 17,
        video_preset: str = "slow",
        video_bitrate: Optional[str] = None,
        video_pixel_format: str = "yuv420p",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._video_quality = video_quality
        self._video_codec = video_codec
        self._video_crf = video_crf
        self._video_preset = video_preset
        self._video_bitrate = video_bitrate
        self._video_pixel_format = video_pixel_format

    @staticmethod
    def _to_video_frame(frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim != 3:
            raise ValueError(f"Expected frame shape (H, W, C), got {frame.shape}")
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.shape[-1] != 3:
            raise ValueError(f"Expected 3 channels, got shape {frame.shape}")
        return np.ascontiguousarray(frame)

    def capture_image(self, infos=None):
        image = super().capture_image(infos=infos)
        if image is None:
            return None
        return self._to_video_frame(image)

    def flush_video(
        self,
        name=None,
        suffix="",
        verbose=False,
        ignore_empty_transition=True,
        save: bool = True,
    ):
        if len(self.render_images) == 0:
            return
        if ignore_empty_transition and len(self.render_images) == 1:
            return
        if save:
            self._video_id += 1
            if name is None:
                video_name = "{}".format(self._video_id)
                if suffix:
                    video_name += "_" + suffix
                if self._avoid_overwriting_video:
                    while (
                        Path(self.output_dir)
                        / (video_name.replace(" ", "_").replace("\n", "_") + ".mp4")
                    ).exists():
                        self._video_id += 1
                        video_name = "{}".format(self._video_id)
                        if suffix:
                            video_name += "_" + suffix
            else:
                video_name = name
            output_path = (
                Path(self.output_dir)
                / (video_name.replace(" ", "_").replace("\n", "_") + ".mp4")
            )
            writer_kwargs = dict(
                fps=self.video_fps,
                codec=self._video_codec,
                pixelformat=self._video_pixel_format,
                macro_block_size=1,
                ffmpeg_log_level="error",
            )
            if self._video_bitrate is not None:
                writer_kwargs["bitrate"] = self._video_bitrate
            elif self._video_quality is not None:
                writer_kwargs["quality"] = self._video_quality
            output_params = ["-preset", self._video_preset, "-movflags", "+faststart"]
            if self._video_crf is not None and self._video_bitrate is None:
                output_params += ["-crf", str(self._video_crf)]
            writer_kwargs["output_params"] = output_params

            writer = imageio.get_writer(str(output_path), **writer_kwargs)
            frames_iter = tqdm.tqdm(self.render_images) if verbose else self.render_images
            for frame in frames_iter:
                writer.append_data(self._to_video_frame(frame))
            writer.close()
            if verbose:
                print(f"Video created: {output_path}")
        self._video_steps = 0
        self.render_images = []

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.evaluate and args.num_iterations == 0:
        # Evaluation should run at least once even if total_timesteps is very small.
        args.num_iterations = 1
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    runs_root = Path(__file__).resolve().parents[3] / "runs"
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Optional auto-resume: find newest checkpoint inside this run directory.
    if args.resume_latest and args.checkpoint is None:
        candidate_paths = []
        final_ckpt = run_dir / "final_ckpt.pt"
        if final_ckpt.exists():
            candidate_paths.append(final_ckpt)
        candidate_paths.extend(sorted(run_dir.glob("ckpt_*.pt")))
        if candidate_paths:
            latest_ckpt = max(candidate_paths, key=lambda p: p.stat().st_mtime)
            args.checkpoint = str(latest_ckpt)
            print(f"Resuming from latest checkpoint: {args.checkpoint}")
        else:
            print(f"--resume-latest set but no checkpoint found under {run_dir}. Starting from scratch.")

    if args.evaluate and not args.checkpoint:
        raise ValueError("--evaluate requires --checkpoint to be set (or use --resume-latest with --exp-name)")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    use_async_train = args.sim_backend == "physx_cpu" and (args.num_envs > 1 and not args.evaluate)
    if args.sim_backend == "physx_cpu" and args.num_eval_envs > 1:
        raise ValueError(
            "num_eval_envs > 1 with physx_cpu is not supported in this script yet. Use --num-eval-envs 1."
        )
    if use_async_train and args.save_train_video_freq is not None:
        raise ValueError(
            "save_train_video_freq is not supported with async CPU vectorization. Set it to None."
        )

    obs_mode = "rgb+depth" if args.include_depth else "rgb"
    env_kwargs = dict(
        obs_mode=obs_mode,
        render_mode=args.render_mode,
        sim_backend=args.sim_backend,
        enable_shadow=args.enable_shadow,
        human_render_camera_configs=dict(
            width=args.render_width,
            height=args.render_height,
            shader_pack=args.render_shader,
        ),
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    eval_envs = FlattenRGBDObservationWrapper(
        eval_envs, rgb=True, depth=args.include_depth, state=args.include_state
    )
    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    if use_async_train:
        def _make_train_worker(worker_idx):
            def _thunk():
                worker_env = gym.make(
                    args.env_id,
                    reconfiguration_freq=args.reconfiguration_freq,
                    **env_kwargs,
                )
                worker_env = FlattenRGBDObservationWrapper(
                    worker_env, rgb=True, depth=args.include_depth, state=args.include_state
                )
                if isinstance(worker_env.action_space, gym.spaces.Dict):
                    worker_env = FlattenActionSpaceWrapper(worker_env)
                return AsyncCPUWorkerEnvAdapter(worker_env)
            return _thunk

        worker_fns = [_make_train_worker(i) for i in range(args.num_envs)]
        envs = AsyncCPUVectorAdapter(
            AsyncVectorEnv(worker_fns, shared_memory=False),
            device=device,
        )
    else:
        envs = gym.make(
            args.env_id,
            num_envs=args.num_envs if not args.evaluate else 1,
            reconfiguration_freq=args.reconfiguration_freq,
            **env_kwargs,
        )
        envs = FlattenRGBDObservationWrapper(
            envs, rgb=True, depth=args.include_depth, state=args.include_state
        )
        if isinstance(envs.action_space, gym.spaces.Dict):
            envs = FlattenActionSpaceWrapper(envs)
    eval_output_dir = None
    if args.capture_video:
        eval_output_dir = str(run_dir / "videos")
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None and not use_async_train:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = ConfiguredRecordEpisode(
                envs,
                output_dir=str(run_dir / "train_videos"),
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=args.video_fps,
                video_quality=args.video_quality,
                video_codec=args.video_codec,
                video_crf=args.video_crf,
                video_preset=args.video_preset,
                video_bitrate=args.video_bitrate,
                video_pixel_format=args.video_pixel_format,
            )
        eval_envs = ConfiguredRecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.evaluate,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=args.video_fps,
            video_quality=args.video_quality,
            video_codec=args.video_codec,
            video_crf=args.video_crf,
            video_preset=args.video_preset,
            video_bitrate=args.video_bitrate,
            video_pixel_format=args.video_pixel_format,
        )
    if not use_async_train:
        envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    if use_async_train:
        env_spec = gym.spec(args.env_id)
        max_episode_steps = env_spec.max_episode_steps if env_spec is not None else 0
    else:
        max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["ppo", "walltime_efficient"]
            )
        writer = SummaryWriter(str(run_dir))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # ALGO Logic: Storage setup
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    agent = Agent(envs, sample_obs=next_obs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        try:
            agent.load_state_dict(torch.load(args.checkpoint))
            print(f"Loaded checkpoint: {args.checkpoint}")
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load checkpoint '{args.checkpoint}'. "
                "This usually means the checkpoint was produced by a different script/model architecture. "
                "Use --exp-name for a run generated by this script (train_ppo_vt.py), or pass a compatible --checkpoint."
            ) from e

    cumulative_times = defaultdict(float)

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if args.eval_during_train and (iteration % args.eval_freq == 1):
            print("Evaluating")
            stime = time.perf_counter()
            try:
                eval_obs, _ = eval_envs.reset()
                eval_metrics = defaultdict(list)
                num_episodes = 0
                for _ in range(args.num_eval_steps):
                    with torch.no_grad():
                        eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                        if "final_info" in eval_infos:
                            mask = eval_infos["_final_info"]
                            num_episodes += mask.sum()
                            for k, v in eval_infos["final_info"]["episode"].items():
                                eval_metrics[k].append(v)
                print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
                for k, v in eval_metrics.items():
                    mean = torch.stack(v).float().mean()
                    if logger is not None:
                        logger.add_scalar(f"eval/{k}", mean, global_step)
                    print(f"eval_{k}_mean={mean}")
                if logger is not None:
                    eval_time = time.perf_counter() - stime
                    cumulative_times["eval_time"] += eval_time
                    logger.add_scalar("time/eval_time", eval_time, global_step)
                if args.evaluate:
                    break
            except RuntimeError as e:
                if "ErrorDeviceLost" in str(e) or "DeviceLost" in str(e):
                    print(
                        "Evaluation hit Vulkan device-lost on macOS. "
                        "Rerun training with --no-eval-during-train and evaluate in a separate run."
                    )
                    if args.evaluate:
                        raise
                    args.eval_during_train = False
                else:
                    raise
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = str(run_dir / f"ckpt_{iteration}.pt")
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        rollout_time = time.perf_counter()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            if (not use_async_train) and ("final_info" in infos):
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"]).view(-1)
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.perf_counter()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
    if args.save_model and not args.evaluate:
        model_path = str(run_dir / "final_ckpt.pt")
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    eval_envs.close()
    if args.capture_video and eval_output_dir is not None:
        video_dir = Path(eval_output_dir)
        if video_dir.exists():
            mp4s = sorted(video_dir.glob("*.mp4"))
            print(f"Video output dir: {video_dir} (mp4 files: {len(mp4s)})")
            if len(mp4s) > 0:
                print(f"Latest video: {mp4s[-1]}")
        else:
            print(f"Video output dir does not exist yet: {video_dir}")
    if logger is not None: logger.close()
