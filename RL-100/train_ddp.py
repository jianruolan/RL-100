if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
import os
import fcntl
import hydra
import torch
import dill
import inspect
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
from copy import deepcopy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import time
import threading
from hydra.core.hydra_config import HydraConfig

# DDP imports
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from rl_100.policy.rl100_3d import RL1003D
from rl_100.policy.rl100_2d import RL1002D
from rl_100.dataset.base_dataset import BaseDataset
from rl_100.env_runner.base_runner import BaseRunner
from rl_100.common.checkpoint_util import TopKCheckpointManager
from rl_100.common.pytorch_util import dict_apply, optimizer_to
from rl_100.model.diffusion.ema_model import EMAModel
from rl_100.model.common.lr_scheduler import get_scheduler
from rl_100.model.common.cm_util import update_ema
from rl_100.unidpg.dynamics_eval_batch import train_dynamics
from rl_100.unidpg.uni_ppo import BehaviorProximalPolicyOptimization
from rl_100.unidpg.critic import IQL_Q_V_no, ValueLearner
from collections import deque
import glob
OmegaConf.register_new_resolver("eval", eval, replace=True)

_IQLFT_RESTORE_RNG = os.environ.get("IQLFT_RESTORE_RNG_AFTER_IQL", "1") == "1"
_IQLFT_RESTORE_RNG_AT_CONSTRUCT = (
    _IQLFT_RESTORE_RNG
    or os.environ.get("IQLFT_RESTORE_RNG_AT_CONSTRUCT", "0") == "1"
)


def _iqlft_snapshot_rng():
    snap = {
        "torch_cpu": torch.random.get_rng_state().clone(),
        "numpy": np.random.get_state(legacy=True),
        "random": random.getstate(),
    }
    if torch.cuda.is_available():
        snap["cuda_all"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    return snap


def _iqlft_restore_rng(snap):
    torch.random.set_rng_state(snap["torch_cpu"])
    if torch.cuda.is_available() and "cuda_all" in snap:
        torch.cuda.set_rng_state_all(snap["cuda_all"])
    np.random.set_state(snap["numpy"])
    random.setstate(snap["random"])


import warnings
warnings.filterwarnings("ignore")
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import pprint
os.environ["WANDB_CONSOLE"] = "off"  # Or "silent" to suppress more messages
os.environ["WANDB_SILENT"] = "true"

def init_wandb_run(cfg, output_dir):
    logging_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
    init_timeout = int(logging_cfg.pop("init_timeout", 120))
    retry_init_timeout = int(logging_cfg.pop("retry_init_timeout", max(init_timeout * 2, 300)))
    settings_cfg = logging_cfg.pop("settings", {}) or {}

    def _wandb_init(timeout):
        settings = dict(settings_cfg)
        settings["init_timeout"] = timeout
        return wandb.init(
            dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(**settings),
            **logging_cfg
        )

    try:
        return _wandb_init(init_timeout)
    except wandb.errors.CommError as exc:
        if "timeout" not in str(exc).lower():
            raise
        cprint(
            f"[WandB] init timed out after {init_timeout}s, retrying with {retry_init_timeout}s",
            "yellow",
        )
        return _wandb_init(retry_init_timeout)

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state dict keys if present (from DDP)"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def _copy_to_cpu(state_dict):
    """Copy state dict tensors to CPU for async saving."""
    return {k: v.cpu().clone() for k, v in state_dict.items()}

class DDPPolicyWrapper:
    """Wrapper to handle DDP models for unio4"""
    def __init__(self, ddp_model):
        self._ddp_model = ddp_model
        self._module = ddp_model.module if hasattr(ddp_model, 'module') else ddp_model

    def __getattr__(self, name):
        # Avoid infinite recursion for special methods
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        # First try to get from module (for accessing model attributes)
        if hasattr(self._module, name):
            return getattr(self._module, name)
        # Then try to get from ddp_model (for forward pass, etc.)
        if hasattr(self._ddp_model, name):
            return getattr(self._ddp_model, name)

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def forward(self, *args, **kwargs):
        return self._ddp_model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._ddp_model(*args, **kwargs)

    def parameters(self):
        return self._ddp_model.parameters()

    def train(self, mode=True):
        return self._ddp_model.train(mode)

    def eval(self):
        return self._ddp_model.eval()

    @property
    def model(self):
        return self._module.model

    @property
    def obs_encoder(self):
        return self._module.obs_encoder

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = ('model_module',)  # Exclude model_module as it's just a reference

    def __init__(self, cfg: OmegaConf, output_dir=None):
        cfg.ppo.num_inference_steps = cfg.policy.num_inference_steps
        if getattr(cfg.ppo, 'iql_adv', False):
            raise ValueError(
                'ppo.iql_adv=True is no longer supported after removing '
                'dp_align_update_iql_no_share; use the default PPO path instead.'
            )
        self.cfg = cfg
        # self.cfg.task.env_runner.seed = self.cfg.training.seed
        self._output_dir = output_dir
        self._saving_thread = None
        self.shm_manager = None  # Initialize shared memory manager

        # DDP setup
        self.is_ddp = dist.is_available() and dist.is_initialized()
        if self.is_ddp:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = torch.device(f'cuda:{self.rank}')
            # Adjust seed for each rank
            seed = cfg.training.seed + self.rank
            # Only print from rank 0
            if self.rank == 0:
                print('Training workspace initialized 1')
        else:
            self.rank = 0
            self.world_size = 1
            self.device = torch.device(cfg.training.device)
            seed = cfg.training.seed
            if self.rank == 0:
                print('Training workspace initialized 1')

        # set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set output_dir early for checkpoint path resolution
        self.output_dir = self.output_dir()

        # configure model
        # Determine if we're using 2D or 3D policy based on config
        policy_target = cfg.policy.get('_target_', '')
        if 'RL1002D' in policy_target or 'rl100_2d' in policy_target:
            self.model: RL1002D = hydra.utils.instantiate(cfg.policy)
            self.is_2d_policy = True
        else:
            self.model: RL1003D = hydra.utils.instantiate(cfg.policy)
            self.is_2d_policy = False

        self.ema_model = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except Exception: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        # Initialize unio4 for both 2D and 3D policies (before DDP wrapping)
        self.unio4 = BehaviorProximalPolicyOptimization(
            policy=self.model,
            device=self.device,
            policy_lr=cfg.unio4.bppo_lr,
            clip_ratio=cfg.unio4.clip_ratio,
            entropy_weight=cfg.unio4.entropy_weight,
            decay=cfg.unio4.decay,
            omega=cfg.unio4.omega,
            batch_size=cfg.unio4.bppo_batch_size,
            is_iql=cfg.critic.is_iql,
            temperature=cfg.unio4.temperature,
            ratio_strategy=cfg.unio4.ratio_strategy,
            top_k=cfg.unio4.top_k,
            num_inference_steps=cfg.policy.num_inference_steps,
            fix_encoder=cfg.unio4.fix_encoder,
            cfg=cfg,
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0
        # Note: Optimizer will be created in run() after DDP wrapping to ensure proper initialization

        if self.rank == 0:
            print('Training workspace initialized 2')

        # Note: Pretrained encoder loading will be handled after DDP setup in run() method

    def get_stage1_artifact_dir(self):
        return self.cfg.unio4.get('stage1_resume_dir', None) or self.output_dir

    def get_critic_artifact_dir(self):
        """Return the directory for critic/value/encoder artifacts.
        When chunk_as_single_action uses a stride-specific critic directory,
        keep critic artifacts separate from the shared stage1 BC/dynamics dir.
        This applies to both offline training and later online loading."""
        stage1_dir = self.get_stage1_artifact_dir()
        if self.cfg.chunk_as_single_action:
            explicit_dir = self.cfg.unio4.get('critic_artifact_dir', None)
            if explicit_dir:
                return explicit_dir

            inferred_dir = os.path.join(
                stage1_dir,
                f'critic_c{self.cfg.n_action_steps}_f{self.cfg.n_action_steps}',
            )
            if os.path.exists(os.path.join(inferred_dir, 'Q_bc_20.pt')):
                return inferred_dir
        return stage1_dir

    def get_stage1_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.get_stage1_artifact_dir()).joinpath('checkpoints', f'{tag}.ckpt')

    def get_global_best_dir(self):
        return self.cfg.unio4.get('global_best_dir', None) or os.path.join(self.output_dir, 'best')

    def get_global_best_ema_dir(self):
        return self.cfg.unio4.get('global_best_ema_dir', None) or os.path.join(self.output_dir, 'best_ema')

    def get_global_best_score_path(self):
        return os.path.join(self.get_global_best_dir(), 'best_score.csv')

    def get_global_best_ema_score_path(self):
        return os.path.join(self.get_global_best_ema_dir(), 'best_score.csv')

    def get_global_best_lock_path(self):
        best_dir = self.get_global_best_dir()
        return os.path.join(os.path.dirname(best_dir), '.global_best.lock')

    def get_global_best_ema_lock_path(self):
        best_dir = self.get_global_best_ema_dir()
        return os.path.join(os.path.dirname(best_dir), '.global_best_ema.lock')

    def _read_best_score(self, score_path):
        if os.path.exists(score_path):
            best_score = np.loadtxt(score_path, delimiter=',')
            if isinstance(best_score, np.ndarray):
                best_score = float(np.asarray(best_score).reshape(-1)[0])
            else:
                best_score = float(best_score)
            return best_score
        return float('-inf')

    def _maybe_update_best(self, score, best_dir, best_score_path, lock_path, save_fn, eval_name):
        os.makedirs(os.path.dirname(best_dir), exist_ok=True)
        with open(lock_path, 'a+') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            best_saved_scores = self._read_best_score(best_score_path)
            is_updated = score > best_saved_scores
            if is_updated:
                os.makedirs(best_dir, exist_ok=True)
                save_fn(best_dir)
                np.savetxt(best_score_path, [score], fmt='%f', delimiter=',')
                meta_path = os.path.join(best_dir, 'best_meta.txt')
                with open(meta_path, 'w') as f:
                    f.write(f"score: {score}\n")
                    f.write(f"eval_name: {eval_name}\n")
                    f.write(f"source_run_dir: {self.output_dir}\n")
                    f.write(f"timestamp_dir: {self.unio4_output_dir}\n")
                    f.write(f"seed: {self.cfg.training.seed}\n")
                    f.write(f"rollout_length: {self.cfg.unio4.rollout_length}\n")
                    f.write(f"bppo_lr: {self.cfg.unio4.bppo_lr}\n")
            else:
                score = best_saved_scores
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return score, is_updated

    def maybe_update_global_best(self, score):
        return self._maybe_update_best(
            score=score,
            best_dir=self.get_global_best_dir(),
            best_score_path=self.get_global_best_score_path(),
            lock_path=self.get_global_best_lock_path(),
            save_fn=self.unio4.save,
            eval_name='Policy Eval',
        )

    def maybe_update_global_best_ema(self, score):
        if self.ema_model is None:
            return float('-inf'), False
        return self._maybe_update_best(
            score=score,
            best_dir=self.get_global_best_ema_dir(),
            best_score_path=self.get_global_best_ema_score_path(),
            lock_path=self.get_global_best_ema_lock_path(),
            save_fn=self.ema_model.save,
            eval_name='EMA Eval',
        )

    def get_online_best_ema_dir(self):
        return os.path.join(self.output_dir, 'online_best_ema')

    def get_online_best_ema_score_path(self):
        return os.path.join(self.get_online_best_ema_dir(), 'best_score.csv')

    def get_online_best_ema_lock_path(self):
        return os.path.join(os.path.dirname(self.get_online_best_ema_dir()), '.online_best_ema.lock')

    def maybe_update_online_best_ema(self, score):
        if self.ema_model is None:
            return float('-inf'), False
        return self._maybe_update_best(
            score=score,
            best_dir=self.get_online_best_ema_dir(),
            best_score_path=self.get_online_best_ema_score_path(),
            lock_path=self.get_online_best_ema_lock_path(),
            save_fn=self.ema_model.save,
            eval_name='Online EMA Eval',
        )

    def run(self):
        # args = parse_args()
        cfg = copy.deepcopy(self.cfg)

        # Flow mode: validate distill_phase
        if cfg.distill_phase is not None and getattr(self.model, 'is_flow', False):
            if cfg.distill_phase not in ('after_dp', 'after_offline', 'online'):
                raise RuntimeError(f"Unsupported distill_phase='{cfg.distill_phase}' for flow mode.")

        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False

        RUN_VALIDATION = False # reduce time cost

        self.unio4_output_dir = os.path.join(self.output_dir, time.strftime("%Y-%m-%d-%H-%M-%S"))
        # save config
        config = vars(cfg)

        def write_dict(f, d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    f.write(f"{' ' * indent}{key}:\n")
                    write_dict(f, value, indent + 4)
                else:
                    f.write(f"{' ' * indent}{key:20} : {value}\n")

        os.makedirs(self.unio4_output_dir, exist_ok=True)
        self.unio4.set_ratio_log_dir(os.path.join(self.unio4_output_dir, 'ratio_logs'))
        config_path = os.path.join(self.unio4_output_dir, 'config.txt')

        with open(config_path, 'w') as f:
            write_dict(f, config)
        if self.rank == 0:
            print('====================================Here==================================')
        # Note: checkpoint loading will happen after optimizer creation
        # device transfer
        device = self.device
        # configure dataset
        dataset: BaseDataset

        # Check if we're using shared memory for DDP
        # Can be controlled via config or environment variable
        use_shared_memory_config = cfg.get('use_shared_memory', False)  # Default to False for compatibility
        use_shared_memory_env = os.environ.get('DISABLE_SHARED_MEMORY', '').lower() != 'true'
        use_shared_memory = self.is_ddp and use_shared_memory_config and use_shared_memory_env

        # Initialize shm_manager to None
        self.shm_manager = None

        if use_shared_memory and self.rank == 0:
            # Only rank 0 sets up shared memory
            from rl_100.common.shared_memory_utils import setup_shared_memory_dataset

            # First, let's resolve the dataset config to avoid interpolation issues
            try:
                dataset_cfg_dict = OmegaConf.to_container(cfg.task.dataset, resolve=True)
                zarr_path = dataset_cfg_dict.get('zarr_path')
                dataset_target = dataset_cfg_dict.get('_target_', '')
            except Exception as e:
                print(f"[Rank 0] Error resolving dataset config: {e}")
                print("[Rank 0] Falling back to regular dataset loading")
                use_shared_memory = False
                zarr_path = None
                dataset_target = ''

            if zarr_path and use_shared_memory:
                try:
                    # Determine which keys to load based on dataset type
                    if 'cloth_head' in dataset_target.lower() or 'ClothHead' in dataset_target:
                        # ClothHead only needs specific keys
                        load_keys = ['state', 'action', 'rgb_head', 'next_rgb_head', 'next_state', 'next_action', 'reward', 'done', 'timeout', 'return']
                    elif 'cloth' in dataset_target.lower() and 'cloth_head' not in dataset_target.lower():
                        # Cloth dataset - only load keys that actually exist in the data file
                        # Note: hand RGB data is not available in current dataset
                        load_keys = ['state', 'action', 'rgb_head', 'rgb_right_hand', 'rgb_left_hand', 'next_state', 'next_action', 'reward', 'done', 'timeout', 'return']
                    else:
                        # Regular Cloth dataset needs all keys
                        load_keys = None

                    # Setup shared memory
                    print(f"[Rank 0] Setting up shared memory for dataset...")
                    info_path, self.shm_manager = setup_shared_memory_dataset(zarr_path, keys=load_keys)

                    # Save info path for other ranks
                    info_path_file = os.path.join(self.output_dir, 'shared_memory_info_path.txt')
                    os.makedirs(self.output_dir, exist_ok=True)
                    with open(info_path_file, 'w') as f:
                        f.write(info_path)
                    print(f"[Rank 0] Shared memory setup complete. Info saved to: {info_path}")
                except Exception as e:
                    print(f"[Rank 0] Error setting up shared memory: {e}")
                    print("[Rank 0] Falling back to regular dataset loading")
                    # If shared memory setup failed, create a marker file
                    info_path_file = os.path.join(self.output_dir, 'shared_memory_info_path.txt')
                    os.makedirs(self.output_dir, exist_ok=True)
                    with open(info_path_file, 'w') as f:
                        f.write("DISABLED")
                    use_shared_memory = False
            else:
                # If shared memory setup failed, create a marker file
                info_path_file = os.path.join(self.output_dir, 'shared_memory_info_path.txt')
                os.makedirs(self.output_dir, exist_ok=True)
                with open(info_path_file, 'w') as f:
                    f.write("DISABLED")
                use_shared_memory = False

        # Synchronize all processes
        if self.is_ddp:
            print(f"[Rank {self.rank}] Waiting at barrier for shared memory setup...")
            # Add small delay for non-rank-0 processes to reduce CPU usage
            if self.rank != 0:
                time.sleep(1)
            dist.barrier()
            print(f"[Rank {self.rank}] Barrier passed, proceeding with dataset loading...")

        if use_shared_memory:
            # All ranks read the info path
            info_path_file = os.path.join(self.output_dir, 'shared_memory_info_path.txt')
            if os.path.exists(info_path_file):
                with open(info_path_file, 'r') as f:
                    info_path = f.read().strip()

                # Check if shared memory was disabled by rank 0
                if info_path == "DISABLED":
                    use_shared_memory = False
                    print(f"[Rank {self.rank}] Shared memory disabled, using regular dataset loading")
                else:
                    # Update config to use shared memory dataset
                    try:
                        dataset_cfg = OmegaConf.to_container(cfg.task.dataset, resolve=True)

                        # Determine which shared memory dataset to use based on original target
                        original_target = dataset_cfg.get('_target_', '')
                        if 'cloth_head' in original_target.lower() or 'ClothHead' in original_target:
                            dataset_cfg['_target_'] = 'rl_100.dataset.cloth_head_shared.ClothHeadShared'
                        elif 'cloth' in original_target.lower() and 'cloth_head' not in original_target.lower():
                            # Only use ClothShared for regular cloth dataset
                            dataset_cfg['_target_'] = 'rl_100.dataset.cloth_shared.ClothShared'
                        else:
                            # For other datasets (like Adroit), skip shared memory
                            print(f"[Rank {self.rank}] Dataset {original_target} does not support shared memory, using regular loading")
                            use_shared_memory = False

                        if use_shared_memory:
                            dataset_cfg['shared_memory_info_path'] = info_path

                            # Create dataset with shared memory
                            print(f"[Rank {self.rank}] Creating shared memory dataset with target: {dataset_cfg['_target_']}")
                            print(f"[Rank {self.rank}] Shared memory info path: {dataset_cfg['shared_memory_info_path']}")
                            dataset = hydra.utils.instantiate(dataset_cfg)
                            print(f"[Rank {self.rank}] Using shared memory dataset")
                        else:
                            # Regular dataset loading without shared memory
                            dataset = hydra.utils.instantiate(cfg.task.dataset)
                            print(f"[Rank {self.rank}] Using regular dataset loading")
                    except Exception as e:
                        print(f"[Rank {self.rank}] Error creating shared memory dataset: {e}")
                        use_shared_memory = False
            else:
                use_shared_memory = False

        if not use_shared_memory:
            # Regular dataset loading
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            print(f"[Rank {self.rank}] Using regular dataset loading")

        # Get shape info for both 2D and 3D policies
        self.dataset = dataset  # Save reference for finetuning dataloader
        self.shape_info = dataset.get_shape_info(self.cfg.horizon - self.model.start, self.cfg.n_obs_steps)
        # import pdb
        # pdb.set_trace()
        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")

        def _safe_dataloader_cfg(base_cfg, batch_size=None, shuffle=None, drop_last=None):
            cfg_dict = OmegaConf.to_container(base_cfg)
            if batch_size is not None:
                cfg_dict['batch_size'] = batch_size
            if shuffle is not None:
                cfg_dict['shuffle'] = shuffle
            if drop_last is not None:
                cfg_dict['drop_last'] = drop_last
            # Point-cloud/image batches are large nested dicts. Pinned-memory copies are
            # brittle here and can fail with `CUDA error: invalid argument` in the
            # pin-memory thread, so keep loaders conservative.
            cfg_dict['pin_memory'] = False
            cfg_dict['persistent_workers'] = False
            cfg_dict['num_workers'] = min(cfg_dict.get('num_workers', 8), 2)
            cfg_dict.pop('sampler', None)
            return cfg_dict

        # Create distributed samplers if using DDP
        if self.is_ddp:
            # Adjust batch size per GPU to maintain effective batch size
            original_batch_size = cfg.dataloader.batch_size
            per_gpu_batch_size = original_batch_size // self.world_size
            if per_gpu_batch_size == 0:
                per_gpu_batch_size = 1
                if self.rank == 0:
                    print(f"Warning: Original batch size {original_batch_size} is smaller than world size {self.world_size}")
                    print(f"Using batch size 1 per GPU, effective batch size will be {self.world_size}")

            train_sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True)
            # Override shuffle in dataloader config when using DistributedSampler
            train_dataloader_cfg = _safe_dataloader_cfg(
                cfg.dataloader,
                batch_size=per_gpu_batch_size,
                shuffle=False,
                drop_last=True,
            )
            train_dataloader = DataLoader(dataset, sampler=train_sampler, **train_dataloader_cfg)

            if self.rank == 0:
                print(f"DDP Training: Adjusting batch size from {original_batch_size} to {per_gpu_batch_size} per GPU")
                print(f"Effective batch size: {per_gpu_batch_size * self.world_size}")
        else:
            train_dataloader = DataLoader(dataset, **_safe_dataloader_cfg(cfg.dataloader))
        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        if self.is_ddp:
            # Adjust validation batch size per GPU
            original_val_batch_size = cfg.val_dataloader.batch_size
            per_gpu_val_batch_size = original_val_batch_size // self.world_size
            if per_gpu_val_batch_size == 0:
                per_gpu_val_batch_size = 1

            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            val_dataloader_cfg = _safe_dataloader_cfg(
                cfg.val_dataloader,
                batch_size=per_gpu_val_batch_size,
                shuffle=False,
            )
            val_dataloader = DataLoader(val_dataset, sampler=val_sampler, **val_dataloader_cfg)
        else:
            val_dataloader = DataLoader(val_dataset, **_safe_dataloader_cfg(cfg.val_dataloader))
        if (self.cfg.off2off and self.cfg.off2off_no_bc) or self.cfg.use_pre_norm:
            del dataset
            norm_dataset = hydra.utils.instantiate(cfg.task.norm_dataset)
            normalizer = norm_dataset.get_normalizer()
            if self.rank == 0:
                cprint('***********************************reuse the normalizer of pre-dataset***********************************', 'yellow')
                cprint('***********************************reuse the normalizer of pre-dataset***********************************', 'yellow')
                cprint('***********************************reuse the normalizer of pre-dataset***********************************', 'yellow')
        else:
            normalizer = dataset.get_normalizer()

        # Note: all_val_data removed to avoid OOM - now using batched val_dataloader for validation

        # Set normalizer
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # Move model to device before DDP wrapping
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)

        # Wrap model with DDP if using distributed training
        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
            self.model_module = self.model.module
        else:
            self.model_module = self.model

        # Create optimizer AFTER DDP wrapping
        # Note: In DDP, gradients are averaged across GPUs, so learning rate typically doesn't need adjustment
        # However, we provide an option to scale lr if needed for exact reproduction
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer)
        if self.is_ddp and cfg.get('scale_lr_with_batch_size', False):
            # Scale learning rate based on the ratio of effective batch size
            # This is typically NOT needed for DDP as gradients are averaged
            original_lr = optimizer_cfg['lr']
            scaled_lr = original_lr * self.world_size
            optimizer_cfg['lr'] = scaled_lr
            if self.rank == 0:
                print(f"Scaling learning rate from {original_lr} to {scaled_lr} (world_size: {self.world_size})")

        self.optimizer = hydra.utils.instantiate(
            optimizer_cfg, params=self.model.parameters())
        optimizer_to(self.optimizer, device)

        # Resume training - load checkpoint after optimizer is created
        if cfg.training.resume:
            latest_ckpt_path = self.get_stage1_checkpoint_path(tag='latest')
            latest_cm_ckpt_path = self.get_stage1_checkpoint_path(tag='latest_cm')
            if latest_cm_ckpt_path.is_file():
                if self.rank == 0:
                    print(f"Resuming cm model from checkpoint {latest_cm_ckpt_path}")
                # Create teacher/distilled_model sub-modules before loading
                # so checkpoint keys like teacher.* and distilled_model.* are accepted.
                if cfg.distill_phase is not None:
                    self.model_module.set_target()
                self.load_checkpoint(path=latest_cm_ckpt_path)
            elif latest_ckpt_path.is_file():
                if self.rank == 0:
                    print(f"Resuming diffusion model from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)
            elif self.rank == 0:
                print(f"No checkpoint found at {latest_ckpt_path}")

        # Fix optimizer state for lr_scheduler if needed, especially on resume.
        for group in self.optimizer.param_groups:
            if 'initial_lr' not in group:
                group['initial_lr'] = group['lr']

        # Configure lr scheduler AFTER optimizer is created
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs)
                // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        self.env_runner = env_runner
        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)
        wandb_run = None
        if self.cfg.use_wandb and self.rank == 0:
            cfg.logging.name = str(cfg.logging.name)
            if self.rank == 0:
                cprint("-----------------------------", "yellow")
                cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
                cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
                cprint("-----------------------------", "yellow")
            # disable wandb logging
            import logging
            wandb_logger = logging.getLogger("wandb")
            wandb_logger.setLevel(logging.ERROR)
            # configure logging
            wandb_run = init_wandb_run(cfg, self.output_dir)
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                },
                allow_val_change=True
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # Load pretrained encoder if configured (typically for 2D policy)
        latest_path = self.get_stage1_checkpoint_path(tag='latest')
        if not os.path.exists(latest_path) or self.cfg.training.resume == False:
            # load pretrained encoder
            if self.cfg.get('use_pretrained_2DEncoder', False):
                if 'channel' in self.cfg.policy.get('stage1_model_name', ''):
                    if self.rank == 0:
                        print('load pretrained encoder')
                    self.model_module.obs_encoder.load_pretrained_encoder(self.get_pretrained_model_path(self.cfg.policy.stage1_model_name), device=self.device)
                    self.model_module.obs_encoder.switch_to_RL_stages()

        # save batch for sampling
        train_sampling_batch = None
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        # import pdb; pdb.set_trace()
        # training loop
        latest_path = self.get_stage1_checkpoint_path(tag='latest')
        # if self.cfg.off2off:
        #     self.model.obs_encoder.load_state_dict(torch.load(os.path.join(self.output_dir, '2025-04-02-16-01-28', 'score_3999', 'encoder.pt')))
        #     self.model.model.load_state_dict(torch.load(os.path.join(self.output_dir, '2025-04-02-16-01-28', 'score_3999', 'model.pt')))
        #     print('load offline pretrained model successfully from {}'.format(os.path.join(self.output_dir, '2025-04-02-16-01-28', 'score_3999')))
        # ===============================stage 1-1: set for diffusion training ===============================
        if not os.path.exists(latest_path) or self.cfg.training.resume == False or (self.cfg.off2off and not self.cfg.off2off_no_bc):
            # VIB module beta kl anealling
            total_steps = cfg.training.num_epochs * len(train_dataloader)
            if hasattr(self.model_module.obs_encoder if self.is_ddp else self.model.obs_encoder, 'beta_kl'):
                target_beta_kl = (self.model_module.obs_encoder if self.is_ddp else self.model.obs_encoder).beta_kl
            for local_epoch_idx in range(cfg.training.num_epochs):
                # Synchronize at the beginning of each epoch
                if self.is_ddp:
                    dist.barrier()

                # Set epoch for distributed sampler
                if self.is_ddp and hasattr(train_dataloader, 'sampler'):
                    train_dataloader.sampler.set_epoch(local_epoch_idx)

                # KL annealing
                model_obs_encoder = self.model_module.obs_encoder if self.is_ddp else self.model.obs_encoder
                if cfg.kl_annealing and hasattr(model_obs_encoder, 'beta_kl'):
                    progress = local_epoch_idx / max(cfg.training.num_epochs - 1, 1)
                    model_obs_encoder.beta_kl = target_beta_kl * progress

                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()

                # Show progress bar only on rank 0
                if self.rank == 0:
                    tepoch = tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                            leave=False, mininterval=cfg.training.tqdm_interval_sec)
                else:
                    tepoch = train_dataloader

                for batch_idx, batch in enumerate(tepoch):

                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                    # if cfg.policy.use_vib:
                    #     progress = self.global_step / total_steps  # [0,1]
                    #     self.model.beta_kl = cfg.policy.beta_kl * progress
                    # compute loss
                    t1_1 = time.time()
                    # For DDP, we need to call compute_loss on the module
                    if self.is_ddp:
                        raw_loss, loss_dict = self.model.module.compute_loss(batch)
                    else:
                        raw_loss, loss_dict = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        # For DDP, update EMA with the underlying module
                        if self.is_ddp:
                            ema.step(self.model.module)
                        else:
                            ema.step(self.model)
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    if self.rank == 0:
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()

                    if verbose and self.rank == 0:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        # Only log from rank 0
                        if self.cfg.use_wandb and self.rank == 0:
                            wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout - synchronize before evaluation
                if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                    # Synchronize all processes before evaluation
                    if self.is_ddp:
                        dist.barrier()

                    if self.rank == 0:
                        t3 = time.time()
                        # runner_log = env_runner.run(policy, dataset=dataset)
                        runner_log = env_runner.run(policy)
                        t4 = time.time()
                        # print(f"rollout time: {t4-t3:.3f}")
                        # log all
                        step_log.update(runner_log)

                    # Synchronize again after evaluation
                    if self.is_ddp:
                        dist.barrier()



                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                # For DDP, we need to call compute_loss on the module
                                if self.is_ddp:
                                    loss, loss_dict = self.model.module.compute_loss(batch)
                                else:
                                    loss, loss_dict = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']

                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        if self.cfg.no_pre_action:
                            gt_action = gt_action[:, self.cfg.n_obs_steps - 1 :]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                if env_runner is None:
                    step_log['test_mean_score'] = - train_loss

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                    # checkpointing - only on rank 0 to avoid conflicts
                    if self.rank == 0:
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()

                        # sanitize metric names
                        metric_dict = dict()
                        for key, value in step_log.items():
                            new_key = key.replace('/', '_')
                            metric_dict[new_key] = value

                        # We can't copy the last checkpoint here
                        # since save_checkpoint uses threads.
                        # therefore at this point the file might have been empty!
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                        if cfg.only_bc and self.rank == 0:
                            # For set_policy, use the underlying module
                            policy_to_set = self.model_module if self.is_ddp else self.model
                            self.unio4.set_policy(policy_to_set); self.unio4.set_old_policy()
                            os.makedirs(os.path.join(self.output_dir, 'best'), exist_ok=True)
                            self.unio4.save(os.path.join(self.output_dir, 'best'))

                # Save checkpoint every 100 epochs during BC training
                if self.epoch % 400 == 0 and self.rank == 0:
                    epoch_ckpt_path = os.path.join(self.output_dir, 'checkpoints', f'epoch_{self.epoch}.ckpt')
                    self.save_checkpoint(path=epoch_ckpt_path)
                    print(f"Saved checkpoint at epoch {self.epoch}: {epoch_ckpt_path}")


                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                if self.cfg.use_wandb and self.rank == 0:
                    wandb_run.log(step_log, step=self.global_step)
                self.global_step += 1
                self.epoch += 1
                del step_log

                # Synchronize at the end of each epoch
                if self.is_ddp:
                    dist.barrier()
        # log_data = self.eval(eval_times=self.cfg.unio4.eval_times)
        self.train_dataloader = train_dataloader

        # Set paths for distillation and finetuning checkpoints
        self.offline_best_path = self.get_global_best_dir()
        self.offline_last_path = os.path.join(self.output_dir, 'last')

        # =============================== stage 1-1: end diffusion training ===============================
        # Distill to consistency model after BC training (before critic/dynamics training)
        if self.cfg.distill_phase == 'after_dp':
            self.distill2cm(train_dataloader, val_dataloader, wandb_run, env_runner, phase='after_dp')
        # if self.cfg.offline:
        #     print('re-create critic dataset without validation data')
        #     # configure dataset
        #     critic_dataset = hydra.utils.instantiate(cfg.task.critic_dataset)
        #     all_dataloader = DataLoader(critic_dataset, batch_size = critic_dataset.get_length())
        #     for data in all_dataloader:
        #         all_data = data
        #     self.all_data = dict_apply(all_data, lambda x: x.to(device, non_blocking=True))

        #     assert isinstance(critic_dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(critic_dataset)}")
        #     critic_dataloader = DataLoader(critic_dataset, **cfg.dataloader)
        #     critic_normalizer = critic_dataset.get_normalizer()

        # --- Offline dataset role split ---
        # critic_dataset: used for offline IQL/critic training
        # finetune_dataset: used for offline BPPO actor update
        # Both default to the same config as task.dataset (sequence_stride=1),
        # but can be overridden independently for chunk boundary experiments.
        if self.cfg.offline and self.cfg.chunk_as_single_action:
            critic_dataset = hydra.utils.instantiate(cfg.task.critic_dataset)
            if self.rank == 0:
                cprint(f'Critic dataset: {len(critic_dataset)} samples '
                       f'(stride={getattr(critic_dataset, "sequence_stride", 1)})', 'cyan')
        else:
            critic_dataset = None  # use train_dataloader as before

        # Create copy_encoder after DDP setup (following train_ddp.py and train_with2D_ddp.py pattern)
        if self.is_ddp:
            copy_encoder = deepcopy(self.model_module.obs_encoder)
        else:
            copy_encoder = deepcopy(self.model.obs_encoder)

        # Use model_module to access the underlying model in DDP mode
        model_ref = self.model_module if self.is_ddp else self.model
        model_ref.set_critic_normalizer(normalizer)
        self.model.to(device)
        # for Uni-O4 fine-tuning
        iql, Q_bc, value = model_ref.initialize_critic(
            device=device,
            q_hidden_dim=self.cfg.critic.q_hidden_dim,
            q_depth=self.cfg.critic.q_depth,
            q_lr=self.cfg.critic.q_lr,
            target_update_freq=self.cfg.critic.target_update_freq,
            tau=self.cfg.critic.tau,
            gamma=self.cfg.critic.gamma,
            v_hidden_dim=self.cfg.critic.v_hidden_dim,
            v_depth=self.cfg.critic.v_depth,
            v_lr=self.cfg.critic.v_lr,
            omega=self.cfg.critic.omega,
            is_double_q=self.cfg.critic.is_double_q,
            is_iql=self.cfg.critic.is_iql,
            is_share_encoder=self.cfg.critic.is_share_encoder,
            use_action_embed=self.cfg.use_action_embed,
            fix_encoder=cfg.critic.fix_encoder,
            chunk_as_single_action=self.cfg.chunk_as_single_action,
            n_action_steps=self.cfg.n_action_steps,
            use_conv_action_embed=getattr(self.cfg, 'use_conv_action_embed', False),
            conv_hidden_dims=getattr(self.cfg, 'conv_hidden_dims', [128, 256]),
            conv_latent_cz=getattr(self.cfg, 'conv_latent_cz', 32),
            conv_kernel_size=getattr(self.cfg, 'conv_kernel_size', 5),
            conv_n_groups=getattr(self.cfg, 'conv_n_groups', 8),
            action_recon_beta=getattr(self.cfg, 'action_recon_beta', 0.5),
            q_layer_norm=getattr(self.cfg.critic, 'q_layer_norm', False),
            action_embed_layer_norm=getattr(self.cfg.critic, 'action_embed_layer_norm', False),
            action_scale_norm=getattr(self.cfg.critic, 'action_scale_norm', False),
            )
        stage1_artifact_dir = self.get_stage1_artifact_dir()
        critic_artifact_dir = self.get_critic_artifact_dir()
        Q_bc_path = os.path.join(critic_artifact_dir, 'Q_bc_20.pt')
        value_path = os.path.join(critic_artifact_dir, 'value_20.pt')
        if self.cfg.critic.is_iql:
            # Q_bc training
            if self.cfg.critic.is_share_encoder:
                encoder_path = os.path.join(critic_artifact_dir, 'encoder.pt')
            else:
                encoder_path = None
            if os.path.exists(Q_bc_path):
                if self.cfg.critic.load_pretrain:
                    iql.load(Q_bc_path, value_path, encoder_path)
                    iql.eval()
                    iql.obs_encoder.eval()
            if not os.path.exists(Q_bc_path) or self.cfg.off2off:
                # Select dataloader for IQL training:
                # Use critic_dataset if available (offline chunk mode), else train_dataloader
                if critic_dataset is not None:
                    critic_dl_cfg = _safe_dataloader_cfg(cfg.dataloader)
                    if self.is_ddp:
                        critic_sampler = DistributedSampler(critic_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True)
                        critic_dl_cfg['shuffle'] = False
                        critic_dl_cfg.pop('sampler', None)
                        critic_dataloader = DataLoader(critic_dataset, sampler=critic_sampler, **critic_dl_cfg)
                    else:
                        critic_dataloader = DataLoader(critic_dataset, **critic_dl_cfg)
                else:
                    critic_dataloader = train_dataloader

                # DDP training for IQL
                if self.is_ddp:
                    # Wrap the critic networks with DDP to enable gradient synchronization.
                    iql._Q = DDP(iql._Q, device_ids=[self.rank], find_unused_parameters=True)
                    iql._value = DDP(iql._value, device_ids=[self.rank], find_unused_parameters=True)

                    # Re-initialize optimizers with DDP model parameters.
                    from torch.optim import Adam
                    iql._q_optimizer = Adam(iql._Q.parameters(), lr=self.cfg.critic.q_lr)
                    iql._v_optimizer = Adam(iql._value.parameters(), lr=self.cfg.critic.v_lr)

                for local_epoch_idx in range(cfg.training.num_critic_epochs):
                    # Synchronize at the beginning of each epoch
                    if self.is_ddp:
                        dist.barrier()

                    # Set epoch for distributed sampler
                    if self.is_ddp and hasattr(critic_dataloader, 'sampler'):
                        critic_dataloader.sampler.set_epoch(local_epoch_idx)

                    critic_step_log = dict()
                    # ========= train for this epoch ==========
                    q_train_losses, v_train_losses = list(), list()

                    # Show progress bar only on rank 0
                    if self.rank == 0:
                        tepoch = tqdm.tqdm(critic_dataloader, desc=f"Training Q_bc epoch {local_epoch_idx}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec)
                    else:
                        tepoch = critic_dataloader

                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        Q_bc_loss, value_loss = iql.update(batch=batch)
                        # Ensure we're storing scalar values, not tensors or numpy arrays
                        if isinstance(Q_bc_loss, torch.Tensor):
                            Q_bc_loss = Q_bc_loss.item()
                        elif isinstance(Q_bc_loss, np.ndarray):
                            Q_bc_loss = float(Q_bc_loss)

                        if isinstance(value_loss, torch.Tensor):
                            value_loss = value_loss.item()
                        elif isinstance(value_loss, np.ndarray):
                            value_loss = float(value_loss)

                        q_train_losses.append(Q_bc_loss)
                        v_train_losses.append(value_loss)

                    # Gather losses from all ranks
                    if self.is_ddp:
                        # Convert lists to tensors
                        q_loss_tensor = torch.tensor(q_train_losses, device=device)
                        v_loss_tensor = torch.tensor(v_train_losses, device=device)

                        # Gather all losses
                        gathered_q_losses = [torch.zeros_like(q_loss_tensor) for _ in range(self.world_size)]
                        gathered_v_losses = [torch.zeros_like(v_loss_tensor) for _ in range(self.world_size)]

                        dist.all_gather(gathered_q_losses, q_loss_tensor)
                        dist.all_gather(gathered_v_losses, v_loss_tensor)

                        # Concatenate and compute mean
                        all_q_losses = torch.cat(gathered_q_losses).cpu().numpy()
                        all_v_losses = torch.cat(gathered_v_losses).cpu().numpy()
                        q_loss_mean = np.mean(all_q_losses)
                        v_loss_mean = np.mean(all_v_losses)
                    else:
                        q_loss_mean, v_loss_mean = np.mean(q_train_losses), np.mean(v_train_losses)

                    if self.rank == 0:
                        print('Step: {}, Q loss: {}, Value loss: {}'.format(local_epoch_idx, q_loss_mean, v_loss_mean))
                        if self.cfg.use_wandb:
                            wandb_run.log({'Q_loss': q_loss_mean, 'value_loss': v_loss_mean})

                # Save model only on rank 0
                if self.rank == 0:
                    os.makedirs(critic_artifact_dir, exist_ok=True)
                    iql.save(Q_bc_path, value_path, encoder_path)

                # Synchronize all ranks after IQL training
                if self.is_ddp:
                    dist.barrier()
            q_eval = iql.minQ
        # load dynamics parameters
        # For DDP, use the underlying module
        prediction_mode = getattr(self.cfg.dynamics, 'prediction_mode', 'last')
        if self.cfg.chunk_as_single_action and prediction_mode != "full":
            raise ValueError(
                "chunk_as_single_action=True requires dynamics.prediction_mode='full'. "
                "A chunk dynamics step advances the whole observation window, so "
                "'last' mode would mix stale observations with the predicted chunk endpoint."
            )
        dynamics_encoder = self.model_module.get_dynamics_encoder() if self.is_ddp else self.model.get_dynamics_encoder()
        if self.cfg.dynamics_type=="diffusion":
            dynamics_path = os.path.join(stage1_artifact_dir, f'saved_models_diffusion_{prediction_mode}')
        else:
            dynamics_path = os.path.join(stage1_artifact_dir, f'saved_models_{prediction_mode}')
        # set dynamics parameters
        obs_feature_dim = self.model_module.obs_feature_dim if self.is_ddp else self.model.obs_feature_dim
        action_dim = self.model_module.action_dim if self.is_ddp else self.model.action_dim
        normalizer = self.model_module.normalizer if self.is_ddp else self.model.normalizer
        if prediction_mode == "full":
            output_obs_dim = obs_feature_dim * self.cfg.n_obs_steps
        else:
            output_obs_dim = obs_feature_dim
        self.cfg.lddm.encoder_output_dim = output_obs_dim
        if getattr(self.cfg, 'use_conv_action_embed', False):
            from rl_100.model.action_ae import ActionChunkEncoder
            conv_encoder = ActionChunkEncoder(
                action_dim=action_dim,
                hidden_dims=list(getattr(self.cfg, 'conv_hidden_dims', [128, 256])),
                latent_cz=getattr(self.cfg, 'conv_latent_cz', 32),
                kernel_size=getattr(self.cfg, 'conv_kernel_size', 5),
                n_groups=getattr(self.cfg, 'conv_n_groups', 8),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, self.cfg.n_action_steps, action_dim)
                self.cfg.lddm.action_embed_dim = conv_encoder(dummy).reshape(1, -1).shape[-1]
        elif prediction_mode == "full":
            self.cfg.lddm.action_embed_dim = obs_feature_dim

        # Create dynamics model on all ranks for DDP training
        # Only rank 0 should handle logging and directory creation
        if self.rank == 0:
            dynamics = train_dynamics(
                env_runner.env,
                normalizer,
                dynamics_encoder,
                dynamics_path,
                self.cfg,
                obs_feature_dim,
                action_dim,
                chunk_as_single_action=self.cfg.chunk_as_single_action,
                n_action_steps=self.cfg.n_action_steps,
                n_obs_steps=self.cfg.n_obs_steps,
                device=device,
            )
        else:
            # For other ranks, create dynamics without logger to avoid directory conflicts
            from rl_100.unidpg.transition_model.models.dynamics_model import EnsembleDynamicsModel
            from rl_100.unidpg.transition_model.models.ensemble_diffusion_dynamics import EnsembleDiffusionDynamicsModel
            from rl_100.unidpg.transition_model.dynamics import EnsembleDynamics_batch
            from rl_100.unidpg.transition_model.utils.termination_fns import get_termination_fn

            if self.cfg.chunk_as_single_action:
                action_dim = action_dim * self.cfg.n_action_steps

            if self.cfg.dynamics_type == 'diffusion':
                lddm_model = hydra.utils.instantiate(self.cfg.lddm)
                lddm_model.set_device(device)

                use_true_ensemble = getattr(self.cfg.dynamics, 'use_true_ensemble', False)
                dynamics_model = EnsembleDiffusionDynamicsModel(
                    lddm_model=lddm_model,
                    obs_dim=output_obs_dim,
                    action_dim=action_dim,
                    num_ensemble=self.cfg.dynamics.n_ensemble,
                    num_elites=self.cfg.dynamics.n_elites,
                    with_reward=self.cfg.predict_r,
                    device=device,
                    use_true_ensemble=use_true_ensemble,
                    cfg=self.cfg,
                )
            else:
                dynamics_model = EnsembleDynamicsModel(
                    obs_dim=output_obs_dim,
                    action_dim=action_dim,
                    hidden_dims=self.cfg.dynamics.dynamics_hidden_dims,
                    num_ensemble=self.cfg.dynamics.n_ensemble,
                    num_elites=self.cfg.dynamics.n_elites,
                    weight_decays=self.cfg.dynamics.dynamics_weight_decay,
                    device=device,
                    cfg=self.cfg,
                    with_reward=self.cfg.predict_r,
                )

            if not self.cfg.dynamics.fix_encoder:
                dynamics_optim = hydra.utils.instantiate(
                    self.cfg.optimizer, params=list(dynamics_model.parameters()) + list(dynamics_encoder.parameters()))
            else:
                for param in dynamics_encoder.parameters():
                    param.requires_grad = False
                dynamics_optim = hydra.utils.instantiate(
                    self.cfg.optimizer, params=dynamics_model.parameters())

            termination_fn = get_termination_fn(task=self.cfg.task_name)
            dynamics = EnsembleDynamics_batch(
                dynamics_model,
                dynamics_optim,
                termination_fn,
                env_runner.env,
                normalizer,
                dynamics_encoder,
                cfg=self.cfg,
                action_dim=action_dim,
                gamma=self.cfg.critic.gamma,
                device=device,
                chunk_as_single_action=self.cfg.chunk_as_single_action,
                n_action_steps=self.cfg.n_action_steps,
                prediction_mode=prediction_mode,
            )
            # Don't set logger for non-rank-0 processes

        # Synchronize after creating dynamics
        if self.is_ddp:
            dist.barrier()

        # Wrap dynamics model with DDP if needed
        if self.is_ddp and hasattr(dynamics, 'model'):
            # Wrap the entire dynamics model with DDP for robust distributed training.
            dynamics.model = DDP(
                dynamics.model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True
            )

            # Re-create the optimizer with the parameters of the DDP-wrapped model.
            if not self.cfg.dynamics.fix_encoder:
                params_to_optimize = list(dynamics.model.parameters()) + list(dynamics_encoder.parameters())
            else:
                params_to_optimize = dynamics.model.parameters()

            dynamics.optim = hydra.utils.instantiate(
                self.cfg.optimizer, params=params_to_optimize
            )

        # DDP training for dynamics
        if ((not os.path.exists(os.path.join(dynamics_path, "dynamics.pth")) and cfg.offline) or self.cfg.off2off):
            epoch = 0
            for local_epoch_idx in range(cfg.dynamics.dynamics_max_epochs):
                # Synchronize at the beginning of each epoch
                if self.is_ddp:
                    dist.barrier()
                    # Set epoch for distributed sampler
                    if hasattr(train_dataloader, 'sampler'):
                        train_dataloader.sampler.set_epoch(local_epoch_idx)

                dynamics_losses = list()
                # Show progress bar only on rank 0
                if self.rank == 0:
                    tepoch = tqdm.tqdm(train_dataloader, desc=f"Training dynamics epoch {local_epoch_idx}",
                            leave=False, mininterval=cfg.dynamics.tqdm_interval_sec)
                else:
                    tepoch = train_dataloader

                epoch += 1
                for batch_idx, batch in enumerate(tepoch):

                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if self.cfg.chunk_as_single_action:
                        nobs_features, next_nobs_features = dynamics.obs2latent(batch['obs']), dynamics.next_obs2latent(batch['next_obs'])
                        single_nob_features, single_next_nob_features = nobs_features[:, -1, :], next_nobs_features[:, -1, :]
                    else:
                        nobs_features, next_nobs_features = dynamics.obs2latent(batch['obs']), dynamics.obs2latent(batch['next_obs'])
                        single_nob_features, single_next_nob_features = nobs_features[:, -1, :], next_nobs_features[:, -1, :]

                    if prediction_mode == "full":
                        batch_size = nobs_features.shape[0]
                        train_nobs = nobs_features.reshape(batch_size, -1)  # [B, n_obs_steps * feature_dim]
                        train_next_nobs = next_nobs_features.reshape(batch_size, -1)
                    else:
                        train_nobs = single_nob_features
                        train_next_nobs = single_next_nob_features

                    dynamics_loss = dynamics.learn(batch=batch, nobs_features=train_nobs, next_nobs_features=train_next_nobs)
                    dynamics.optimize(dynamics_loss)

                    if isinstance(dynamics_loss, torch.Tensor):
                        dynamics_loss = dynamics_loss.item()
                    elif isinstance(dynamics_loss, np.ndarray):
                        dynamics_loss = float(dynamics_loss)
                    dynamics_losses.append(dynamics_loss)

                # Gather losses from all ranks
                if self.is_ddp:
                    loss_tensor = torch.tensor(dynamics_losses, device=device)
                    gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(self.world_size)]
                    dist.all_gather(gathered_losses, loss_tensor)
                    all_losses = torch.cat(gathered_losses).cpu().numpy()
                    loss_mean = np.mean(all_losses)
                else:
                    loss_mean = np.mean(dynamics_losses)

                if (local_epoch_idx + 1) % 10 == 0 and self.rank == 0:
                    print('dynamics loss: {}'.format(loss_mean))

                # Batched validation (only on rank 0) to avoid OOM
                should_stop = torch.tensor([0], device=device)
                if self.rank == 0:
                    with torch.no_grad():
                        val_losses_all = []
                        for val_batch in val_dataloader:
                            val_batch = dict_apply(val_batch, lambda x: x.to(device, non_blocking=True))
                            if self.cfg.chunk_as_single_action:
                                val_nobs_features, val_next_nobs_features = dynamics.obs2latent(val_batch['obs']), dynamics.next_obs2latent(val_batch['next_obs'])
                                val_single_nob_features, val_single_next_nob_features = val_nobs_features[:, -1, :], val_next_nobs_features[:, -1, :]
                            else:
                                val_nobs_features, val_next_nobs_features = dynamics.obs2latent(val_batch['obs']), dynamics.obs2latent(val_batch['next_obs'])
                                val_single_nob_features, val_single_next_nob_features = val_nobs_features[:, -1, :], val_next_nobs_features[:, -1, :]

                            if prediction_mode == "full":
                                batch_size = val_nobs_features.shape[0]
                                val_nobs = val_nobs_features.reshape(batch_size, -1)
                                val_next_nobs = val_next_nobs_features.reshape(batch_size, -1)
                            else:
                                val_nobs = val_single_nob_features
                                val_next_nobs = val_single_next_nob_features

                            # Compute validation loss for this batch
                            val_inputs, val_targets = dynamics.format_samples_for_training(val_batch, val_nobs, val_next_nobs)
                            batch_val_losses = dynamics.validate(val_inputs, val_targets)
                            val_losses_all.append(batch_val_losses)

                        # Aggregate validation losses across batches
                        val_losses_all = np.array(val_losses_all)  # [num_batches, num_ensemble]
                        new_holdout_losses = val_losses_all.mean(axis=0).tolist()  # [num_ensemble]

                    # Update holdout losses and early stopping logic
                    dynamics._update_holdout_and_log(new_holdout_losses, np.mean(dynamics_losses), wandb_run if wandb_run is not None else wandb, epoch)

                    # Early stopping check (same logic as train.py)
                    if (dynamics.cnt >= cfg.dynamics.max_epochs_since_update) or (cfg.dynamics.dynamics_max_epochs and (epoch >= cfg.dynamics.dynamics_max_epochs)):
                        print(f'Early stopping at epoch {epoch}')
                        should_stop = torch.tensor([1], device=device)

                # Synchronize early stopping decision across all ranks
                if self.is_ddp:
                    dist.broadcast(should_stop, src=0)

                if should_stop.item() == 1:
                    break

                # Synchronize after validation
                if self.is_ddp:
                    dist.barrier()

            # Save dynamics model after training (only on rank 0)
            if self.rank == 0:
                dynamics.post_well_learned()
                # The wrapper handles unwrapping automatically in state_dict()
                dynamics.save(dynamics_path)

        # Synchronize all ranks after dynamics training
        if self.is_ddp:
            dist.barrier()

        # All ranks load dynamics
        if cfg.offline:
            if os.path.exists(os.path.join(dynamics_path, "dynamics.pth")):
                # Load the saved dynamics model
                dynamics.load(dynamics_path)

        #===============================================Stage 2 finetune dp3 by unio4 offline===============================================
        # For set_policy, use the DDP-wrapped model to enable gradient synchronization.
        if self.is_ddp:
            policy_to_set = DDPPolicyWrapper(self.model)
        else:
            policy_to_set = self.model
        self.unio4.set_policy(policy_to_set); self.unio4.set_old_policy()
        if cfg.eval:
            if self.cfg.unio4.idql_eval:
                log_data = self.unio4_eval(
                    idql_eval = True,
                    dynamics = dynamics,
                    first_action = self.cfg.unio4.first_action,
                    get_np = True,
                    iql = iql,
                    Q = Q_bc,
                    repeat_num = 128,
                    eval_times=self.cfg.unio4.eval_times
                )
            else:
                log_data = self.eval(eval_times=self.cfg.unio4.eval_times)

            score = log_data['test_mean_score']
            return score
        else:
            if cfg.offline:
                self.finetune_dp3(dynamics, Q_bc, value, iql, wandb, ema)

        #===============================================Stage 2-2 distill to cm from finetuned diffusion=================================================
        if self.cfg.distill_phase == 'after_offline':
            # Load the offline finetuned model
            offline_cp_timestamp = getattr(self.cfg, 'offline_cp_timestamp', None)
            offline_cp_timestep = getattr(self.cfg, 'offline_cp_timestep', None)
            print(f"offline_cp_timestamp: {offline_cp_timestamp}, offline_cp_timestep: {offline_cp_timestep}")
            if offline_cp_timestamp and offline_cp_timestep:
                self.unio4.load(os.path.join(self.output_dir, offline_cp_timestamp, offline_cp_timestep))
            else:
                self.unio4.load(os.path.join(self.offline_best_path))
            self.distill2cm(train_dataloader, val_dataloader, wandb_run, env_runner, phase='after_offline')

        #===============================================Stage 3 finetune dp3 by unio4 online=================================================
        if cfg.online:
            if self.is_ddp:
                if self.cfg.distill_phase == 'online' and not self.cfg.ppo.load_online_cp:
                    offline_distilled_path = os.path.join(self.offline_best_path, 'last', 'distilled_model.pt')
                    if os.path.exists(offline_distilled_path):
                        if self.rank == 0:
                            cprint(
                                'found offline distilled model for online distill: {}'.format(
                                    offline_distilled_path
                                ),
                                'green'
                            )
                    else:
                        if self.rank == 0:
                            cprint(
                                'offline distilled model not found at {}; running offline distill before online'.format(
                                    offline_distilled_path
                                ),
                                'yellow'
                            )
                        self.distill2cm(
                            train_dataloader,
                            val_dataloader,
                            wandb_run,
                            env_runner,
                            phase='after_offline'
                        )
                        if not os.path.exists(offline_distilled_path):
                            raise RuntimeError(
                                'Offline distill completed but did not create '
                                f'{offline_distilled_path}'
                            )

                    # DDP mode in train_ddp.py only prepares the offline distilled
                    # artifact. The launcher then starts a second single-GPU process
                    # for online RL (see scripts/Flow/Online/3D/train_policy_online_flow_chunk.sh).
                    if self.rank == 0:
                        cprint(
                            '[DDP Distill] Offline distill completed. Exiting current process; '
                            'start the online stage with single-GPU train.py.',
                            'green'
                        )
                    dist.barrier()
                    return

                raise RuntimeError(
                    'train_ddp.py does not support executing the online RL stage under DDP. '
                    'Use DDP here for BC/offline/distill only, then launch the single-GPU '
                    'online stage via train.py or scripts/Flow/Online/3D/train_policy_online_flow_chunk.sh.'
                )

            model_ref = self.model

            if self.cfg.load_bc:
                if self.cfg.distill_phase == 'online' and not self.cfg.ppo.load_online_cp:
                    raise RuntimeError(
                        "distill_phase='online' requires an offline policy checkpoint. "
                        "Disable load_bc or resume from an online checkpoint."
                    )
                self.unio4._policy.model.load_state_dict(model_ref.model.state_dict())
                self.unio4._policy.obs_encoder.load_state_dict(model_ref.obs_encoder.state_dict())
                self.unio4.set_old_policy()
            elif self.cfg.distill_phase in ('after_dp', 'after_offline'):
                # distill2cm() already ran on the correct model and promoted the student.
                # For after_dp: load promoted student from offline_best_path/last/
                # For after_offline: self.unio4._policy was distilled+promoted, skip reload.
                if self.cfg.distill_phase == 'after_dp':
                    self.unio4.load(os.path.join(self.offline_best_path, 'last'))

                # Fix flow_inference_steps: ppo.load() / set_old_policy() don't persist it.
                if getattr(self.unio4._policy, 'is_flow', False):
                    target_steps = self.unio4._policy.flow_distill_inference_steps
                    self.unio4._policy.flow_inference_steps = target_steps
                    self.unio4._old_policy.flow_inference_steps = target_steps
                    cprint(f'restored flow_inference_steps={target_steps} for promoted model (policy + old_policy)', 'yellow')
            else:
                if self.cfg.offline_cp_timestamp and self.cfg.offline_cp_timestep is not None:
                    self.unio4.load(os.path.join(self.output_dir, self.cfg.offline_cp_timestamp, self.cfg.offline_cp_timestep))
                else:
                    self.unio4.load(os.path.join(self.offline_best_path))

            if cfg.ppo.iql_ft:
                _iqlft_construct_snapshot = (
                    _iqlft_snapshot_rng() if _IQLFT_RESTORE_RNG_AT_CONSTRUCT else None
                )
                online_iql_encoder = deepcopy(model_ref.obs_encoder)
                online_iql_share_encoder = self.cfg.ppo.is_share_iql_encoder
                online_iql_fix_encoder = self.cfg.ppo.fix_iql_encoder
                iql_online = IQL_Q_V_no(
                device=self.device,
                state_dim=model_ref.obs_feature_dim * model_ref.n_obs_steps,
                feature_dim=model_ref.obs_feature_dim,
                action_dim=model_ref.action_dim,
                q_hidden_dim=self.cfg.critic.q_hidden_dim,
                q_depth=self.cfg.critic.q_depth,
                Q_lr=self.cfg.ppo.ft_q_lr,
                target_update_freq=self.cfg.critic.target_update_freq,
                tau=self.cfg.critic.tau,
                gamma=self.cfg.critic.gamma,
                v_hidden_dim=self.cfg.critic.v_hidden_dim,
                v_depth=self.cfg.critic.v_depth,
                v_lr=self.cfg.ppo.ft_v_lr,
                omega=self.cfg.ppo.iql_omega,
                is_double_q=self.cfg.critic.is_double_q,
                dp3_normalizer=model_ref.normalizer,
                obs_encoder=online_iql_encoder,
                n_obs_steps=model_ref.n_obs_steps,
                is_share_encoder=online_iql_share_encoder,
                use_pc_color=model_ref.use_pc_color,
                use_action_embed=self.cfg.use_action_embed,
                fix_encoder=online_iql_fix_encoder,
                encoder_update_with=self.cfg.ppo.iql_encoder_update_with,
                n_action_steps=self.cfg.n_action_steps,
                chunk_as_single_action=self.cfg.chunk_as_single_action,
                use_conv_action_embed=getattr(self.cfg, 'use_conv_action_embed', False),
                conv_hidden_dims=getattr(self.cfg, 'conv_hidden_dims', [128, 256]),
                conv_latent_cz=getattr(self.cfg, 'conv_latent_cz', 32),
                conv_kernel_size=getattr(self.cfg, 'conv_kernel_size', 5),
                conv_n_groups=getattr(self.cfg, 'conv_n_groups', 8),
                action_recon_beta=getattr(self.cfg, 'action_recon_beta', 0.5),
                q_layer_norm=getattr(self.cfg.critic, 'q_layer_norm', False),
                action_embed_layer_norm=getattr(self.cfg.critic, 'action_embed_layer_norm', False),
                action_scale_norm=getattr(self.cfg.critic, 'action_scale_norm', False),
                )
                iql_online.eval_with_raw_obs = True
                if os.path.exists(Q_bc_path):
                    if self.cfg.critic.load_pretrain:
                        online_encoder_path = encoder_path if encoder_path and os.path.exists(encoder_path) else None
                        if not online_iql_share_encoder:
                            iql_online.load_with_encoder(Q_bc_path, value_path, online_encoder_path)
                        else:
                            iql_online.load(
                                Q_bc_path,
                                value_path,
                                online_encoder_path,
                                force_load=online_encoder_path is not None,
                            )
                        cprint('load Q_bc and value for online iql finetuning successfully', 'green')
                if _IQLFT_RESTORE_RNG_AT_CONSTRUCT:
                    _iqlft_restore_rng(_iqlft_construct_snapshot)
            else:
                iql_online = None
            self.online_ft(dynamics, Q_bc, value, iql, iql_online, copy_encoder, wandb, ema)

    def finetune_dp3(self, dynamics, Q, value, iql, wandb, ema):
        # evaluation for dp3 pretrained by bc

        policy_refs = [self.model, self.unio4._policy, self.unio4._old_policy]
        aug_restore = []
        disabled_aug = False
        offline_use_aug = getattr(self.cfg, 'offline_use_aug', False)
        if not offline_use_aug:
            for policy_ref in policy_refs:
                if hasattr(policy_ref, 'module'):
                    policy_ref = policy_ref.module
                if hasattr(policy_ref, 'use_aug'):
                    aug_restore.append((policy_ref, policy_ref.use_aug))
                    if policy_ref.use_aug:
                        policy_ref.use_aug = False
                        disabled_aug = True
            if disabled_aug and self.rank == 0:
                print('Disabled image augmentation for offline RL finetuning stage')

        # Create a new dataloader with configurable batch size for finetuning
        # Use finetune_dataset if available (offline chunk mode), else fall back to self.dataset
        if self.cfg.offline and self.cfg.chunk_as_single_action and hasattr(self.cfg.task, 'finetune_dataset'):
            finetune_dataset = hydra.utils.instantiate(self.cfg.task.finetune_dataset)
            if self.rank == 0:
                cprint(f'Finetune dataset: {len(finetune_dataset)} samples '
                       f'(stride={getattr(finetune_dataset, "sequence_stride", 1)})', 'cyan')
        else:
            finetune_dataset = self.dataset

        finetune_batch_size = getattr(self.cfg.unio4, 'finetune_batch_size', self.cfg.dataloader.batch_size)
        finetune_dataloader_cfg = OmegaConf.to_container(self.cfg.dataloader)
        # Finetune batches carry obs, next_obs, rewards and returns, so the host-side
        # batch is much larger than BC batches. Pinned-memory copies for these nested
        # batches are brittle on single-GPU sweep jobs and can fail with
        # `CUDA error: invalid argument` in the pin-memory thread.
        finetune_dataloader_cfg['pin_memory'] = False
        finetune_dataloader_cfg['persistent_workers'] = False
        finetune_dataloader_cfg['num_workers'] = min(finetune_dataloader_cfg.get('num_workers', 8), 2)

        if self.is_ddp:
            # In DDP mode, divide batch size by world_size (same as train_dataloader)
            per_gpu_batch_size = finetune_batch_size // self.world_size
            finetune_dataloader_cfg['batch_size'] = per_gpu_batch_size
            # Create DistributedSampler for DDP
            finetune_sampler = DistributedSampler(finetune_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True)
            finetune_dataloader_cfg['shuffle'] = False  # Shuffle is handled by sampler
            finetune_dataloader_cfg.pop('sampler', None)
            self.finetune_dataloader = DataLoader(finetune_dataset, sampler=finetune_sampler, **finetune_dataloader_cfg)
            if self.rank == 0:
                print(f'Finetuning with total batch size: {finetune_batch_size}, per GPU: {per_gpu_batch_size}')
        else:
            finetune_dataloader_cfg['batch_size'] = finetune_batch_size
            self.finetune_dataloader = DataLoader(finetune_dataset, **finetune_dataloader_cfg)
            print(f'Finetuning with batch size: {finetune_batch_size}')

        self._finetune_iter = None  # Will be initialized in sample_finetune_batch

        self.unio4.set_old_policy()
        # Fix encoder in eval mode for the entire offline finetuning stage
        if self.cfg.unio4.fix_encoder:
            self.unio4._policy.obs_encoder.eval()
            self.unio4._old_policy.obs_encoder.eval()
        best_bppo_path = self.unio4_output_dir
        if self.rank == 0:
            os.makedirs(best_bppo_path, exist_ok=True)
        # import pdb; pdb.set_trace()
        best_saved_scores = float('-inf')
        run_idql_eval = bool(self.cfg.unio4.idql_eval)
        if run_idql_eval:
            idql_log_data = self.unio4_eval(
                idql_eval = True,
                dynamics = dynamics,
                first_action = self.cfg.unio4.first_action,
                get_np = True,
                iql = iql,
                Q = Q,
                repeat_num = 128,
                eval_times=self.cfg.unio4.eval_times
            )
        else:
            idql_log_data = None
        normal_log_data = self.eval(eval_times=self.cfg.unio4.eval_times)

        # Use the configured eval mode for initial best score
        if run_idql_eval:
            log_data = idql_log_data
        else:
            log_data = normal_log_data

        best_bppo_scores = log_data['test_mean_score']
        # offline policy evaluation for dp3 pretrained by bc
        if self.rank == 0:
            best_saved_scores, is_updated = self.maybe_update_global_best(best_bppo_scores)
            if is_updated:
                print('------------saved best model----------------')

        best_mean_qs = dynamics.rollout(
            self.unio4._policy,
            Q,
            iql,
            self.sample_finetune_batch(),
            rollout_length=self.cfg.unio4.rollout_length,
            is_iql=self.cfg.critic.is_iql,
            use_gae=self.cfg.unio4.use_gae,
            first_action = self.cfg.dynamics.first_action,
        )
        if self.rank == 0:
            print('rollout trajectory q mean:{}'.format(best_mean_qs))
        update_num = 0
        success_num = 0
        current_bppo_score = 0
        scores, opes = [], []
        idql_scores, normal_scores = [], []
        scores.append(best_bppo_scores)
        if run_idql_eval:
            idql_scores.append(idql_log_data['test_mean_score'])
        normal_scores.append(normal_log_data['test_mean_score'])
        opes.append(best_mean_qs[0].detach().cpu().numpy())
        if self.rank == 0:
            init_log_data = {
                'current_bppo_scores': best_bppo_scores,
                'current_mean_qs': best_mean_qs,
                'normal_eval_scores': normal_log_data['test_mean_score']
            }
            if run_idql_eval:
                init_log_data['idql_eval_scores'] = idql_log_data['test_mean_score']
            wandb.log(init_log_data)
        # Show progress bar only on rank 0
        if self.rank == 0:
            iterator = tqdm.tqdm(range(int(self.cfg.unio4.bppo_steps)), desc='bppo updating ......')
        else:
            iterator = range(int(self.cfg.unio4.bppo_steps))

        for step in iterator:
            # Synchronize at the beginning of each step for DDP
            if self.is_ddp:
                dist.barrier()

            if self.cfg.unio4.is_linear_decay:
                bppo_lr_now = self.cfg.unio4.bppo_lr * (1 - step / self.cfg.unio4.bppo_steps)
                q_lr_now = self.cfg.critic.q_lr * (1 - step / self.cfg.unio4.bppo_steps)
                clip_ratio_now = self.cfg.unio4.clip_ratio * (1 - step / self.cfg.unio4.bppo_steps)
            else:
                bppo_lr_now = None
                q_lr_now = None
                clip_ratio_now = None
            if step > 200:
                self.cfg.unio4.is_clip_decay = False
                self.cfg.unio4.is_bppo_lr_decay = False
            # finetune dp3 by unio4
            losses = self.unio4.update_distribution(
                batch=self.sample_finetune_batch(),
                value=value,
                Q=Q,
                iql=iql,
                is_clip_decay = self.cfg.unio4.is_clip_decay,
                is_lr_decay = self.cfg.unio4.is_bppo_lr_decay,
                is_linear_decay=self.cfg.unio4.is_linear_decay,
                bppo_lr_now= bppo_lr_now,
                clip_ratio_now= clip_ratio_now,
                dynamics=dynamics,
                use_gae=self.cfg.unio4.use_gae,
                fix_encoder=self.cfg.unio4.fix_encoder,
                final_reward=self.cfg.unio4.final_reward,
                gamma=self.cfg.critic.gamma,
                lamda=self.cfg.ppo.lamda,
                )
            if self.cfg.training.use_ema:
                ema.step(self.unio4._policy)
            if self.rank == 0:
                wandb.log({'dpg_loss': losses})
            # evaluation during training
            if (step+1) % self.cfg.unio4.eval_freq == 0:
                if run_idql_eval:
                    idql_log_data = self.unio4_eval(
                        idql_eval = True,
                        dynamics = dynamics,
                        first_action = self.cfg.unio4.first_action,
                        get_np = True,
                        iql = iql,
                        Q = Q,
                        repeat_num = 128,
                        eval_times=self.cfg.unio4.eval_times
                    )
                    idql_current_scores = idql_log_data['test_mean_score']
                    idql_scores.append(idql_current_scores)

                # Run normal eval
                normal_log_data = self.eval(eval_times=self.cfg.unio4.eval_times)
                normal_current_scores = normal_log_data['test_mean_score']
                normal_scores.append(normal_current_scores)

                # Use the configured eval mode for model saving
                if run_idql_eval:
                    log_data = idql_log_data
                    current_bppo_scores = idql_current_scores
                else:
                    log_data = normal_log_data
                    current_bppo_scores = normal_current_scores

                # Only save models and files on rank 0
                if self.rank == 0:
                    best_saved_scores, is_updated = self.maybe_update_global_best(current_bppo_scores)
                    if is_updated:
                        print('------------saved best model----------------')
                    else:
                        os.makedirs(os.path.join(best_bppo_path, 'score_{}'.format(step)), exist_ok=True)
                        self.unio4.save(os.path.join(best_bppo_path, 'score_{}'.format(step)))
                        print('------------saved {} model----------------'.format(current_bppo_scores))
                scores.append(current_bppo_scores)
                if self.rank == 0:
                    if run_idql_eval:
                        print(f"Step: {step}, IDQL Score: {idql_current_scores}, Normal Score: {normal_current_scores}, Selected Score: {current_bppo_scores}")
                    else:
                        print(f"Step: {step}, Normal Score: {normal_current_scores}, Selected Score: {current_bppo_scores}")
                    eval_log_data = {
                        'current_bppo_scores': current_bppo_scores,
                        'normal_eval_scores': normal_current_scores
                    }
                    if run_idql_eval:
                        eval_log_data['idql_eval_scores'] = idql_current_scores
                    wandb.log(eval_log_data)
            # offline policy evaluation to determin whether to update behavior policy
            if (step+1)% self.cfg.unio4.eval_step == 0:
                current_mean_qs = dynamics.rollout(
                    self.unio4._policy,
                    Q,
                    iql,
                    self.sample_finetune_batch(),
                    rollout_length=self.cfg.unio4.rollout_length,
                    is_iql=self.cfg.critic.is_iql,
                    use_gae=self.cfg.unio4.use_gae,
                    first_action = self.cfg.dynamics.first_action,
                )
                if self.rank == 0:
                    wandb.log({'current_mean_qs': current_mean_qs})
                if self.rank == 0:
                    print('rollout trajectory q mean:{}'.format(current_mean_qs))
                    print(f"Step: {step}, Loss: ", losses)
                if self.cfg.unio4.is_update_old_policy:
                    if current_mean_qs > best_mean_qs:
                        best_mean_qs = current_mean_qs
                        self.unio4.set_old_policy()
                        if self.rank == 0:
                            print('------------------------------update behavior policy----------------------------------------')
                opes.append(current_mean_qs[0].detach().cpu().numpy())
                if self.rank == 0:
                    np.savetxt(os.path.join(best_bppo_path, 'each_ope_score.csv'), opes, fmt='%f', delimiter=',')
            if self.rank == 0:
                np.savetxt(os.path.join(best_bppo_path, 'each_scores.csv'), scores, fmt='%f', delimiter=',')
        # Save final results only on rank 0
        if self.rank == 0:
            np.savetxt(os.path.join(best_bppo_path, 'last_ope_score.csv'), opes, fmt='%f', delimiter=',')
            if run_idql_eval and len(idql_scores) > 0:
                np.savetxt(os.path.join(best_bppo_path, 'last_idql_eval_scores.csv'), idql_scores, fmt='%f', delimiter=',')
            if len(normal_scores) > 0:
                np.savetxt(os.path.join(best_bppo_path, 'last_normal_eval_scores.csv'), normal_scores, fmt='%f', delimiter=',')
            os.makedirs(os.path.join(self.output_dir, 'last'), exist_ok=True)
            self.unio4.save(os.path.join(self.output_dir, 'last'))
            self.unio4.flush_ratio_logs(force=True)
        for policy_ref, use_aug in aug_restore:
            policy_ref.use_aug = use_aug
        if disabled_aug and self.rank == 0:
            print('Restored image augmentation setting after offline RL finetuning stage')
        # Synchronize all ranks after finetune
        if self.is_ddp:
            dist.barrier()
        # wandb.finish()

    def get_distill_optimizer(self,):
        cfg = self.cfg
        cm_optimizer = torch.optim.AdamW(
            self.unio4._policy.distilled_model.parameters(),
            lr=cfg.optimizer.lr,
            betas=(cfg.optimizer.betas[0], cfg.optimizer.betas[1]),
            weight_decay=cfg.optimizer.weight_decay,
            eps=cfg.optimizer.eps)

        cm_lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=cm_optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
               cfg.ppo.max_train_steps * cfg.ppo.K_epochs)# \
                    # // cfg.training.gradient_accumulate_every
        )
        return cm_optimizer, cm_lr_scheduler

    def distill2cm(self, train_dataloader, val_dataloader, wandb_run, env_runner, phase: str = 'after_dp'):
        """Distill from diffusion model to consistency model using DDIM solver.

        This method supports DDP training for consistency model distillation.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            wandb_run: Wandb run object for logging
            env_runner: Environment runner for evaluation
            phase: Distillation phase - 'after_dp' (after BC training) or 'after_offline' (after offline finetuning)
        """
        if self.rank == 0:
            cprint('start distill to cm {}'.format(phase), 'green')

        # =============================== stage 1-2: set for distillation to consistency model using ddim solver ===============================
        cfg = self.cfg
        device = self.device

        # Determine which model to optimize based on phase
        if phase == 'after_dp':
            # Use the underlying model (without DDP wrapper) for optimization
            model_to_optimize = self.model_module if self.is_ddp else self.model
        elif phase == 'after_offline':
            model_to_optimize = self.unio4._policy
        else:
            raise ValueError(f"Unknown distillation phase: {phase}")

        # Set up target model and distilled model
        model_to_optimize.set_target()

        # Create optimizer for distilled model
        cm_optimizer = torch.optim.AdamW(
            model_to_optimize.distilled_model.parameters(),
            lr=cfg.optimizer.lr,
            betas=(cfg.optimizer.betas[0], cfg.optimizer.betas[1]),
            weight_decay=cfg.optimizer.weight_decay,
            eps=cfg.optimizer.eps)

        # Create learning rate scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=cm_optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every
        )

        # Configure checkpoint manager
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # Reset training state for distillation
        distill_global_step = 0
        distill_epoch = 0
        train_sampling_batch = None

        # Note: EMA model will be created after DDP wrapping to ensure parameter consistency
        # See below where ema is created inside the should_train block

        # Determine the checkpoint path based on phase
        if phase == 'after_dp':
            latest_cm_path = self.get_checkpoint_path(tag='latest_cm')
        elif phase == 'after_offline':
            latest_cm_path = os.path.join(self.offline_best_path, 'last', 'distilled_model.pt')

        # Check if we need to train or can resume from checkpoint
        should_train = not os.path.exists(latest_cm_path) or cfg.training.resume == True

        if should_train:
            # Wrap distilled_model with DDP if using distributed training
            original_distilled_model = model_to_optimize.distilled_model
            if self.is_ddp:
                model_to_optimize.distilled_model = DDP(
                    model_to_optimize.distilled_model,
                    device_ids=[self.rank],
                    find_unused_parameters=True
                )
                # Recreate optimizer with DDP-wrapped model parameters
                cm_optimizer = torch.optim.AdamW(
                    model_to_optimize.distilled_model.parameters(),
                    lr=cfg.optimizer.lr,
                    betas=(cfg.optimizer.betas[0], cfg.optimizer.betas[1]),
                    weight_decay=cfg.optimizer.weight_decay,
                    eps=cfg.optimizer.eps)
                # Recreate lr scheduler
                lr_scheduler = get_scheduler(
                    cfg.training.lr_scheduler,
                    optimizer=cm_optimizer,
                    num_warmup_steps=cfg.training.lr_warmup_steps,
                    num_training_steps=(
                        len(train_dataloader) * cfg.training.num_epochs) \
                            // cfg.training.gradient_accumulate_every
                )

            # Create EMA model AFTER DDP wrapping to ensure parameter consistency
            # For distillation, we only need EMA of the distilled_model, not the whole model_to_optimize
            # So we create a separate EMA for just the distilled model's underlying parameters
            ema_distilled_model = deepcopy(original_distilled_model)  # Copy the unwrapped distilled model
            ema: EMAModel = None
            if cfg.training.use_ema:
                ema = hydra.utils.instantiate(
                    cfg.ema,
                    model=ema_distilled_model)

            for local_epoch_idx in range(cfg.training.num_epochs):
                # Synchronize at the beginning of each epoch
                if self.is_ddp:
                    dist.barrier()
                    # Set epoch for distributed sampler
                    if hasattr(train_dataloader, 'sampler'):
                        train_dataloader.sampler.set_epoch(local_epoch_idx)

                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()

                # Show progress bar only on rank 0
                if self.rank == 0:
                    tepoch = tqdm.tqdm(train_dataloader, desc=f"[CM Distill] Training epoch {distill_epoch}",
                            leave=False, mininterval=cfg.training.tqdm_interval_sec)
                else:
                    tepoch = train_dataloader

                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss based on distillation loss type
                    t1_1 = time.time()
                    distill_loss_type = getattr(cfg, 'distill_loss_type', 'action')
                    distill2mean = getattr(cfg, 'distill2mean', False)

                    if getattr(model_to_optimize, 'is_flow', False):
                        raw_loss, loss_dict = model_to_optimize.compute_flow_distill_loss(
                            batch, distill2mean=distill2mean)
                    elif distill_loss_type == 'back_up':
                        raw_loss, loss_dict = model_to_optimize.compute_ddim2cm_loss(batch, distill2mean=distill2mean)
                    elif distill_loss_type == 'action':
                        raw_loss, loss_dict = model_to_optimize.compute_ddim2cm_loss_action(batch, distill2mean=distill2mean)
                    elif distill_loss_type == 'action_same_noise':
                        raw_loss, loss_dict = model_to_optimize.compute_ddim2cm_loss_action_same_noise(batch, distill2mean=distill2mean)
                    else:
                        raw_loss, loss_dict = model_to_optimize.compute_ddim2cm_loss_action(batch, distill2mean=distill2mean)

                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    t1_2 = time.time()

                    # step optimizer
                    if distill_global_step % cfg.training.gradient_accumulate_every == 0:
                        # Gradient clipping
                        if self.is_ddp:
                            torch.nn.utils.clip_grad_norm_(model_to_optimize.distilled_model.parameters(), cfg.training.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model_to_optimize.distilled_model.parameters(), cfg.training.max_grad_norm)
                        cm_optimizer.step()
                        lr_scheduler.step()
                        cm_optimizer.zero_grad(set_to_none=True)

                        # Update EMA for target model (skip for flow — no target_model)
                        if not getattr(model_to_optimize, 'is_flow', False):
                            ema_decay = getattr(cfg.training, 'ema_decay', 0.9999)
                            if self.is_ddp:
                                update_ema(model_to_optimize.target_model.parameters(),
                                           model_to_optimize.distilled_model.module.parameters(), ema_decay)
                            else:
                                update_ema(model_to_optimize.target_model.parameters(),
                                           model_to_optimize.distilled_model.parameters(), ema_decay)

                    t1_3 = time.time()

                    # update ema - use the underlying distilled model (not DDP wrapped)
                    if cfg.training.use_ema:
                        # Get the underlying distilled model for EMA update
                        distilled_for_ema = model_to_optimize.distilled_model.module if self.is_ddp else model_to_optimize.distilled_model
                        ema.step(distilled_for_ema)

                    t1_4 = time.time()

                    # logging
                    raw_loss_cpu = raw_loss.item()
                    if self.rank == 0:
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'cm_train_loss': raw_loss_cpu,
                        'cm_global_step': distill_global_step,
                        'cm_epoch': distill_epoch,
                        'cm_lr': lr_scheduler.get_last_lr()[0]
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        if self.cfg.use_wandb and self.rank == 0 and wandb_run is not None:
                            wandb_run.log(step_log, step=distill_global_step)
                        distill_global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['cm_train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = model_to_optimize
                policy.eval()

                # run rollout - synchronize before evaluation
                RUN_ROLLOUT = not cfg.training.debug
                RUN_VALIDATION = False  # reduce time cost

                if (distill_epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                    # Synchronize all processes before evaluation
                    if self.is_ddp:
                        dist.barrier()

                    if self.rank == 0:
                        t3 = time.time()
                        distill2mean = getattr(cfg, 'distill2mean', False)
                        runner_log = env_runner.run(policy, use_cm=True, distill2mean=distill2mean)
                        t4 = time.time()
                        # log all
                        step_log.update(runner_log)

                    # Synchronize again after evaluation
                    if self.is_ddp:
                        dist.barrier()

                # run validation
                if (distill_epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                    with torch.no_grad():
                        val_losses = list()

                        if self.rank == 0:
                            val_tepoch = tqdm.tqdm(val_dataloader, desc=f"[CM Distill] Validation epoch {distill_epoch}",
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec)
                        else:
                            val_tepoch = val_dataloader

                        for batch_idx, batch in enumerate(val_tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss, loss_dict = model_to_optimize.compute_loss(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['cm_val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (distill_epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']

                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        if getattr(cfg, 'no_pre_action', False):
                            gt_action = gt_action[:, cfg.n_obs_steps - 1 :]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['cm_train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                if env_runner is None:
                    step_log['test_mean_score'] = - train_loss

                # checkpoint
                if (distill_epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                    # Only save on rank 0
                    if self.rank == 0:
                        if phase == 'after_dp':
                            # checkpointing
                            if cfg.checkpoint.save_last_ckpt:
                                self.save_checkpoint(tag='latest_cm')
                            if cfg.checkpoint.save_last_snapshot:
                                self.save_snapshot(tag='latest_cm')

                            # sanitize metric names
                            metric_dict = dict()
                            for key, value in step_log.items():
                                new_key = key.replace('/', '_')
                                metric_dict[new_key] = value
                                metric_dict['type'] = 'cm'

                            # We can't copy the last checkpoint here
                            # since save_checkpoint uses threads.
                            # therefore at this point the file might have been empty!
                            topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                            if topk_ckpt_path is not None:
                                self.save_checkpoint(path=topk_ckpt_path)
                            if getattr(cfg, 'only_bc', False):
                                policy_to_set = self.model_module if self.is_ddp else self.model
                                self.unio4.set_policy(policy_to_set)
                                self.unio4.set_old_policy()
                                os.makedirs(os.path.join(self.output_dir, 'best_cm'), exist_ok=True)
                                self.unio4.save(os.path.join(self.output_dir, 'best_cm'))

                        # Save intermediate checkpoint for both phases
                        os.makedirs(os.path.join(self.offline_best_path, '_{}'.format(str(distill_epoch))), exist_ok=True)
                        # Unwrap DDP model before saving
                        if self.is_ddp and hasattr(model_to_optimize.distilled_model, 'module'):
                            # Temporarily replace with unwrapped model for saving
                            wrapped_model = model_to_optimize.distilled_model
                            model_to_optimize.distilled_model = wrapped_model.module
                            model_to_optimize.save(os.path.join(self.offline_best_path, '_{}'.format(str(distill_epoch))))
                            model_to_optimize.distilled_model = wrapped_model
                        else:
                            model_to_optimize.save(os.path.join(self.offline_best_path, '_{}'.format(str(distill_epoch))))

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                if self.cfg.use_wandb and self.rank == 0 and wandb_run is not None:
                    wandb_run.log(step_log, step=distill_global_step)
                distill_global_step += 1
                distill_epoch += 1
                del step_log

                # Synchronize at the end of each epoch
                if self.is_ddp:
                    dist.barrier()

            # Unwrap DDP model after training loop
            if self.is_ddp:
                model_to_optimize.distilled_model = original_distilled_model
                # Copy state dict from DDP model if it was different
                # (This shouldn't be needed as we're using the same underlying model)

            # Save final checkpoint
            if self.rank == 0:
                if phase == 'after_dp':
                    self.save_checkpoint(tag='latest_cm')
                os.makedirs(os.path.join(self.offline_best_path, 'last'), exist_ok=True)
                model_to_optimize.save(os.path.join(self.offline_best_path, 'last'))

        # After flow distillation: promote student to default model and save promoted checkpoint
        if getattr(model_to_optimize, 'is_flow', False):
            model_to_optimize.promote_distilled_model()
            if self.rank == 0:
                os.makedirs(os.path.join(self.offline_best_path, 'last'), exist_ok=True)
                model_to_optimize.save(os.path.join(self.offline_best_path, 'last'))
                cprint('saved promoted student checkpoint to {}'.format(
                    os.path.join(self.offline_best_path, 'last')), 'green')

        # Synchronize all ranks after distillation
        if self.is_ddp:
            dist.barrier()

        if self.rank == 0:
            cprint('finished distill to cm {}'.format(phase), 'green')
        # =============================== stage 1-2: end distillation training ===============================

    def _prepare_offline_iql_batch_for_online(self, offline_batch):
        """Match offline samples to the online IQL buffer contract."""
        start = self.cfg.n_obs_steps - 1

        if getattr(self.cfg, 'chunk_as_single_action', False):
            end = start + self.cfg.n_action_steps
            action_len = offline_batch['action'].shape[1]
            if action_len < end:
                raise ValueError(
                    f"offline IQL batch action horizon {action_len} is shorter "
                    f"than required chunk slice [{start}:{end}]")

            offline_batch['obs'] = dict_apply(
                offline_batch['obs'],
                lambda x: x[:, :self.cfg.n_obs_steps])
            offline_batch['next_obs'] = dict_apply(
                offline_batch['next_obs'],
                lambda x: x[:, -self.cfg.n_obs_steps:])

            offline_batch['action'] = offline_batch['action'][:, start:end]
            if self.cfg.action_norm:
                offline_batch['action'] = self.model.normalizer['action'].normalize(
                    offline_batch['action'])

            reward_chunk = offline_batch['reward'][:, start:end]
            if reward_chunk.shape[-1] == 1:
                reward_chunk = reward_chunk.squeeze(-1)
            gamma = float(getattr(self.cfg, 'gamma', self.cfg.critic.gamma))
            gamma_weights = torch.pow(
                torch.tensor(gamma, device=reward_chunk.device, dtype=reward_chunk.dtype),
                torch.arange(
                    self.cfg.n_action_steps,
                    device=reward_chunk.device,
                    dtype=reward_chunk.dtype,
                ),
            )
            offline_batch['reward'] = (
                reward_chunk * gamma_weights.reshape(1, -1)
            ).sum(dim=1).reshape(-1, 1, 1)
            offline_batch['not_done'] = offline_batch['not_done'][:, end - 1:end]
            return offline_batch

        offline_batch['action'] = offline_batch['action'][:, start:]
        offline_batch['reward'] = offline_batch['reward'][:, start:]
        offline_batch['not_done'] = offline_batch['not_done'][:, start:]
        if self.cfg.action_norm:
            offline_batch['action'] = self.model.normalizer['action'].normalize(
                offline_batch['action'])
        return offline_batch

    def _next_offline_iql_batch_for_online(self):
        """Reuse the offline dataloader iterator during online IQL updates."""
        offline_iter = getattr(self, '_online_iql_offline_iter', None)
        if offline_iter is None:
            offline_iter = iter(self.train_dataloader)
            self._online_iql_offline_iter = offline_iter

        try:
            offline_batch = next(offline_iter)
        except StopIteration:
            offline_iter = iter(self.train_dataloader)
            self._online_iql_offline_iter = offline_iter
            offline_batch = next(offline_iter)

        offline_batch = dict_apply(
            offline_batch,
            lambda x: x.to(self.device, non_blocking=True))
        return self._prepare_offline_iql_batch_for_online(offline_batch)

    def online_ft(self, dynamics, Q, value, iql, iql_online, copy_encoder, wandb, ema):
        from rl_100.unidpg.online_buffer import ReplayBuffer
        from rl_100.unidpg.online_buffer import IqlBuffer
        use_vec_env = getattr(self.cfg.ppo, 'use_vec_env_online', False)

        # DDP unwrap: the copied body from train.py reads scalar attrs like
        # self.model.obs_encoder / global_cond_dim / normalizer / action_dim.
        # nn.Module.__getattr__ on a DDP wrapper only resolves registered
        # submodules, so those attribute reads must hit the underlying module.
        # Check by type, not by self.is_ddp, because we may have already
        # flipped self.is_ddp=False on rank 0 in run() after tearing down the
        # process group.
        _orig_model = self.model
        if isinstance(self.model, DDP):
            self.model = self.model.module
        _orig_dynamics_model = None
        if dynamics is not None and hasattr(dynamics, 'model') and isinstance(dynamics.model, DDP):
            _orig_dynamics_model = dynamics.model
            dynamics.model = dynamics.model.module
        try:
            return self._online_ft_impl(dynamics, Q, value, iql, iql_online, copy_encoder, wandb, ema, use_vec_env, ReplayBuffer, IqlBuffer)
        finally:
            self.model = _orig_model
            if _orig_dynamics_model is not None:
                dynamics.model = _orig_dynamics_model

    def _online_ft_impl(self, dynamics, Q, value, iql, iql_online, copy_encoder, wandb, ema, use_vec_env, ReplayBuffer, IqlBuffer):
        # VIB: optionally force stochastic sampling in online stage while encoder is in eval mode.
        enable_force_stochastic = getattr(self.cfg.ppo, 'force_stochastic_online', True)

        def _set_force_stochastic(encoder, val):
            if hasattr(encoder, 'force_stochastic'):
                encoder.force_stochastic = val

        def _set_iql_deterministic(iql_ref):
            if iql_ref is None:
                return
            encoders = [getattr(iql_ref, 'obs_encoder', None)]
            for net in [iql_ref._Q, iql_ref._target_Q, iql_ref._value]:
                encoders.append(getattr(net, '_obs_encoder', None))
            for encoder in encoders:
                if encoder is not None:
                    encoder.eval()
                    _set_force_stochastic(encoder, False)

        _set_force_stochastic(self.model.obs_encoder, enable_force_stochastic)
        _set_force_stochastic(self.unio4._policy.obs_encoder, enable_force_stochastic)
        _set_iql_deterministic(iql)
        _set_iql_deterministic(iql_online)
        if self.cfg.distill_phase == 'online':
            self.unio4._policy.set_target()
            distilled_path = os.path.join(self.offline_best_path, 'last/distilled_model.pt')
            if self.cfg.ppo.load_online_cp:
                cprint(
                    'skip offline distilled model load because ppo.load_online_cp=True; '
                    'online checkpoint will restore distilled model',
                    'yellow'
                )
            elif os.path.exists(distilled_path):
                self.unio4._policy.distilled_model.load_state_dict(torch.load(distilled_path))
                print('load distilled model from {} for online distill successfully'.format(distilled_path))
            else:
                raise RuntimeError(
                    "distill_phase='online' requires offline distill first, but "
                    f"{distilled_path} does not exist."
                )
            cm_optimizer, cm_lr_scheduler = self.get_distill_optimizer()
        else:
            cm_optimizer, cm_lr_scheduler = None, None
        online_ft_path = os.path.join(self.output_dir, 'online_ft', time.strftime("%Y-%m-%d-%H-%M-%S"))
        config = vars(self.cfg)

        def write_dict(f, d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    f.write(f"{' ' * indent}{key}:\n")
                    write_dict(f, value, indent + 4)
                else:
                    f.write(f"{' ' * indent}{key:20} : {value}\n")

        os.makedirs(online_ft_path, exist_ok=True)
        config_path = os.path.join(online_ft_path, 'config.txt')

        with open(config_path, 'w') as f:
            write_dict(f, config)

        reward_scaler = None
        if self.cfg.ppo.scale_strategy == 'dynamic' or self.cfg.ppo.scale_strategy == 'number':
            critic_dataset = hydra.utils.instantiate(self.cfg.task.critic_dataset)
            assert isinstance(critic_dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(critic_dataset)}")
            critic_dataloader = DataLoader(critic_dataset, **self.cfg.dataloader)
            # if self.cfg.ppo.share_encoder:
            online_value_encoder = self.unio4._policy.obs_encoder
            # else:
            #     online_value_encoder = copy_encoder
            value = ValueLearner(
                self.device,
                self.model.global_cond_dim,
                self.cfg.critic.v_hidden_dim,
                self.cfg.critic.v_depth,
                self.cfg.critic.v_lr,
                self.model.normalizer,
                online_value_encoder,
                self.model.n_obs_steps,
                self.model.use_pc_color,
                share_encoder=self.cfg.ppo.share_encoder,
                )
            if self.cfg.ppo.share_encoder:
                v_path = os.path.join(self.output_dir, 'value_{}_{}.pt'.format(self.cfg.ppo.scale_strategy, self.cfg.ppo.share_encoder))
            else:
                v_path = os.path.join(self.output_dir, 'value_{}.pt'.format(self.cfg.ppo.scale_strategy))
            from rl_100.unidpg.utils import RewardScaling
            # reward_scaler = RewardScaling(shape=1, gamma=0.99)
            scale_dataset = hydra.utils.instantiate(self.cfg.task.scale_dataset)
            assert isinstance(critic_dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(critic_dataset)}")
            scale_dataloader = DataLoader(scale_dataset, **self.cfg.dataloader)
            reward_scaler = scale_dataset.reward_norm
            cprint('start training value network with dynamic reward scaling', 'green')
            if os.path.exists(v_path):
                value.load(v_path)
            elif self.cfg.ppo.scale_strategy == 'number':
                epoch = 0
                for local_epoch_idx in range(self.cfg.ppo.num_critic_epochs):

                    v_train_losses = list()
                    epoch += 1
                    with tqdm.tqdm(critic_dataloader, desc=f"Training epoch {epoch}",
                                leave=False, mininterval=self.cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch['reward'], batch['return'] = batch['reward'] * 0.1, batch['return'] * 0.1
                            batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                            value_loss = value.update(batch)
                            v_train_losses.append(value_loss)
                    if local_epoch_idx % int(10) == 0:
                        print('Step: {}, Value loss: {}'.format(local_epoch_idx, np.mean(v_train_losses)))
                value.save(v_path)
            elif self.cfg.ppo.scale_strategy == 'dynamic':

                epoch = 0
                for local_epoch_idx in range(self.cfg.ppo.num_critic_epochs):
                    v_train_losses = list()
                    epoch += 1
                    with tqdm.tqdm(scale_dataloader, desc=f"Training epoch {epoch}",
                                leave=False, mininterval=self.cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch['reward'], batch['return'] = batch['reward'], batch['return']
                            batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                            value_loss = value.update(batch)
                            v_train_losses.append(value_loss)
                    if local_epoch_idx % int(10) == 0:
                        print('Step: {}, Value loss: {}'.format(local_epoch_idx, np.mean(v_train_losses)))

                value.save(v_path)

            value_net = value._value
        else:
            value_net = iql.get_online_value_buget(self.cfg)

        # configure env
        # env_runner: BaseRunner
        # env_runner = hydra.utils.instantiate(
        #     self.cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # # TODO: add seed in env's init
        # assert isinstance(env_runner, BaseRunner)
        # Sync inference step config after all distill-phase load/promote logic,
        # before buffer creation. Promoted student uses fewer steps (e.g. 1),
        # so buffer shapes must match the active policy's output.
        if getattr(self.unio4._policy, 'is_flow', False):
            active_steps = self.unio4._policy.flow_inference_steps
            if active_steps != self.cfg.ppo.num_inference_steps:
                cprint(f'syncing num_inference_steps: {self.cfg.ppo.num_inference_steps} -> {active_steps}', 'yellow')
                self.cfg.ppo.num_inference_steps = active_steps
                self.cfg.num_inference_steps = active_steps
                self.cfg.policy.num_inference_steps = active_steps

        replay_buffer = ReplayBuffer(args=self.cfg.ppo, shape_info=self.shape_info, device=self.device)
        if self.cfg.ppo.iql_ft or self.cfg.update_phase == 'outloop':
            iql_buffer = IqlBuffer(None, args=self.cfg.ppo, shape_info=self.shape_info, device=self.device)
            # iql_buffer.initial_with_dataset(self.all_data)
            iql = iql_online
        if self.cfg.ppo.load_online_cp:
            online_cp_path = os.path.join(self.output_dir, 'online_ft')
            dirs = glob.glob(f"{online_cp_path}/*")
            logdir = sorted(dirs)[-1]
            iql, value_net = self.load_online_checkpoints(logdir, iql, value_net, ema)
        self.unio4.transfer2online(critic=value_net, dynamics=dynamics, cfg=self.cfg, cm_optimizer=cm_optimizer, cm_lr_scheduler=cm_lr_scheduler)

        # Sync EMA to current online policy starting point (only for fresh offline→online,
        # NOT when resuming from online checkpoint which already restored EMA)
        if self.cfg.training.use_ema and self.ema_model is not None and ema is not None:
            if not self.cfg.ppo.load_online_cp:
                ema_state = self.ema_model.state_dict()
                policy_state = self.unio4._policy.state_dict()
                filtered_state = {k: v for k, v in policy_state.items() if k in ema_state}
                self.ema_model.load_state_dict(filtered_state, strict=False)
                ema.optimization_step = 0

        if use_vec_env:
            self._online_ft_vec(dynamics, Q, iql, iql_online, wandb, online_ft_path, cm_optimizer, cm_lr_scheduler,
                                ema=ema, reward_scaler_template=reward_scaler if self.cfg.ppo.scale_strategy == 'dynamic' else None)
            return

        # start training and data collection
        total_steps = 0
        env_runner = self.env_runner
        env = env_runner.env
        env.seed(int(self.cfg.training.seed))
        all_success_rates, all_returns = [], []
        cm_all_success_rates, cm_all_returns = [], []
        all_idql_success_rates, all_idql_returns = [], []
        all_ema_success_rates, all_ema_returns = [], []
        if self.cfg.ppo.idql_eval:
            idql_log_data = self.unio4_eval(
                    idql_eval = True,
                    dynamics = dynamics,
                    first_action = self.cfg.unio4.first_action,
                    get_np = True,
                    use_gae=self.cfg.unio4.use_gae,
                    iql = iql,
                    Q = Q,
                    repeat_num = 128,
                    eval_times=self.cfg.unio4.eval_times
                    )
            all_idql_success_rates.append(idql_log_data['test_mean_score'])
            all_idql_returns.append(idql_log_data['mean_returns'])
            log_data = self.eval(eval_times=self.cfg.unio4.eval_times, online=True)
            if self.cfg.distill_phase == 'online':
                cm_log_data = self.eval(
                    online=True, eval_times=self.cfg.unio4.eval_times,
                    use_cm=True, distill2mean=self.cfg.distill2mean)
                cm_all_success_rates.append(cm_log_data['test_mean_score'])
                cm_all_returns.append(cm_log_data['mean_returns'])
            else:
                cm_all_success_rates.append(0)
                cm_all_returns.append(0)
        else:
            log_data = self.eval(eval_times=self.cfg.unio4.eval_times, online=True)
            if self.cfg.distill_phase == 'online':
                cm_log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times, use_cm=True, distill2mean=self.cfg.distill2mean)
                cm_all_success_rates.append(cm_log_data['test_mean_score'])
                cm_all_returns.append(cm_log_data['mean_returns'])
            else:
                cm_all_success_rates.append(0)
                cm_all_returns.append(0)
            all_idql_success_rates.append(0)
            all_idql_returns.append(0)
        all_success_rates.append(log_data['test_mean_score'])
        all_returns.append(log_data['mean_returns'])
        # Initial EMA eval
        ema_log_data = None
        if self.cfg.training.use_ema and self.ema_model is not None:
            ema_log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times,
                                     policy_override=self.ema_model, eval_name='Online EMA Eval')
            all_ema_success_rates.append(ema_log_data['test_mean_score'])
            all_ema_returns.append(ema_log_data['mean_returns'])
            _, is_updated_ema = self.maybe_update_online_best_ema(ema_log_data['test_mean_score'])
            if is_updated_ema:
                print('------------saved online best EMA model----------------')
        else:
            all_ema_success_rates.append(0)
            all_ema_returns.append(0)
        cprint('start online finetuning, initial policy SR: {}, EMA SR: {}'.format(
            log_data['test_mean_score'],
            ema_log_data['test_mean_score'] if ema_log_data else 'N/A'), 'green')
        wandb.log({'online ppo success rates': log_data['test_mean_score'], 'cm success rates': cm_all_success_rates, 'cm returns': cm_all_returns,
                        'online ppo returns': log_data['mean_returns'],
                        'online ema success rates': ema_log_data['test_mean_score'] if ema_log_data else 0,
                        'online ema returns': ema_log_data['mean_returns'] if ema_log_data else 0,})
        # progress_bar = tqdm.tqdm(total=self.cfg.ppo.max_train_steps, desc="Training Progress")
        evaluate_num = 0
        actor_losses, critic_losses, bc_losses, distill_losses = [], [], [], []
        q_train_losses, v_train_losses = [], []
        total_mean_return = []
        total_reward_sub = 0
        total_episode_r =  deque(maxlen=10)
        episode_reward = 0
        time1 = 0
        episode_steps = 0
        update_num = 0
        idql_log_data = None
        while total_steps < self.cfg.ppo.max_train_steps:
            # start rollout
            obs = env.reset()
            # policy.reset()
            done = False
            total_count_sub = 0
            if self.cfg.ppo.scale_strategy == 'dynamic':
                reward_scaler.reset()
            print('episode reward: {}, episode length: {}'.format(episode_reward, episode_steps))
            total_episode_r.append(episode_reward)
            episode_steps = 0
            episode_reward = 0
            # obs['image'] = np.transpose(obs['image'], (0,2,3,1))
            if self.cfg.ppo.clip_std_decay:
                decay_value = self.value_decay(initial_value=self.cfg.clip_std_max, total_steps=total_steps, max_train_steps=self.cfg.ppo.max_train_steps)
                self.unio4._policy.noise_scheduler.clip_std_max = decay_value
            while not done:
                episode_steps += 1
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=self.device))
                # run policy
                obs_dict_input = {}  # flush unused keys
                obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                if 'dexart' in self.cfg.task_name:
                    obs_dict_input['imagin_robot'] = obs_dict['imagin_robot'].unsqueeze(0)
                obs_dict_input['image'] = (obs_dict['image'].unsqueeze(0)).to(torch.float)
                if self.cfg.ppo.idql_rollout:
                    action, all_x, a_logprob = self.unio4._policy.sample_action_with_logprob(obs_dict_input, dynamics=dynamics, first_action=self.cfg.unio4.first_action, use_gae=self.cfg.unio4.use_gae, iql=iql, Q=Q, repeat_num=128)
                else:
                    action, all_x, a_logprob = self.unio4._policy.all_step_action_logprob(obs_dict_input, fix_encoder=self.cfg.ppo.fix_encoder)

                # device_transfer
                all_x = all_x.squeeze(1).detach().to('cpu').numpy()
                a_logprob = a_logprob.squeeze(1).detach().to('cpu').numpy()

                # step env
                next_obs, reward, done, info = env.step(action.squeeze(0).detach().to('cpu').numpy(), reward_agg_method='discounted_sum', gamma=self.cfg.gamma)

                # next_obs['image'] = np.transpose(next_obs['image'], (0,2,3,1))
                if done and episode_steps != self.cfg.task.env_runner.max_steps:
                    dw = True
                else:
                    dw = False
                episode_reward += reward
                # store transition
                obs_dict = dict_apply(obs_dict,
                                      lambda x: x.detach().to('cpu').numpy())
                # next_obs_dict = dict_apply(dict(next_obs), lambda x: x.squeeze())
                if self.cfg.ppo.scale_strategy == 'number':
                    replay_buffer.store(obs_dict, all_x, a_logprob, reward * 0.1, next_obs, done, dw)
                elif self.cfg.ppo.scale_strategy == 'dynamic':
                    scaled_r = reward_scaler(reward)[0]
                    replay_buffer.store(obs_dict, all_x, a_logprob, scaled_r, next_obs, done, dw)
                else:
                    replay_buffer.store(obs_dict, all_x, a_logprob, reward, next_obs, done, dw)

                if self.cfg.ppo.iql_ft or self.cfg.update_phase == 'outloop':
                    iql_buffer.store(obs=obs_dict, action=all_x[-1], reward=reward, next_obs=next_obs, done=done)

                if self.cfg.update_phase == 'outloop':
                    alpha = 0.8 + (1 - 0.8) * (total_steps / self.cfg.ppo.max_train_steps) # linearly increase the alpha from 0.5 to 1
                    idql_bs = int(getattr(self.cfg.ppo, 'idql_batch_size', 256))
                    online_sample_size = int(alpha * idql_bs)
                    offline_sample_size = idql_bs - online_sample_size
                    online_batch = iql_buffer.sample(batch_size=online_sample_size)
                    offline_batch = self.sample_batch(batch_size=offline_sample_size)
                    offline_batch = self._prepare_offline_iql_batch_for_online(offline_batch)
                    merged_batch = iql_buffer.merge(online_batch, offline_batch)
                    distill_loss = self.unio4.distill_update(merged_batch, online=True)
                    distill_losses.append(distill_loss)
                obs = next_obs
                total_steps += 1
                # progress_bar.update(1)
                total_count_sub += 1
                if replay_buffer.count == self.cfg.ppo.batch_size:
                    update_num += 1
                    if self.cfg.ppo.iql_ft:
                        # iql_buffer.store(obs=obs_dict, action=all_x[-1], reward=reward, next_obs=next_obs, done=done)
                        if total_steps > self.cfg.ppo.online_start_training:
                            rng_snapshot = _iqlft_snapshot_rng() if _IQLFT_RESTORE_RNG else None
                            print('start online iql training')
                            for _ in range(self.cfg.ppo.iql_steps):
                                alpha = self.cfg.ppo.data_ratio + (1 - self.cfg.ppo.data_ratio) * (total_steps / self.cfg.ppo.max_train_steps) # linearly increase the alpha from 0.5 to 1
                                idql_bs = int(getattr(self.cfg.ppo, 'idql_batch_size', 256))
                                online_sample_size = int(alpha * idql_bs)
                                offline_sample_size = idql_bs - online_sample_size
                                online_batch = iql_buffer.sample(batch_size=online_sample_size)
                                offline_batch = self._next_offline_iql_batch_for_online()
                                merged_batch = iql_buffer.merge(online_batch, offline_batch)
                                merged_batch = dict_apply(merged_batch, lambda x: x[:idql_bs]) # batch size idql_bs, and online batch is larger
                                Q_bc_loss, value_loss = iql.update(batch=merged_batch, online=True, pre_cut=True, online_recon=self.cfg.ppo.online_iql_recon)
                            if total_steps % self.cfg.ppo.evaluate_freq  == 0:
                                print('Step: {}, Q loss: {}, Value loss: {}'.format(total_steps, Q_bc_loss, value_loss))
                                wandb.log({'online iql Q_loss': Q_bc_loss, 'online iql value value_loss': value_loss})
                            q_train_losses.append(Q_bc_loss); v_train_losses.append(value_loss)
                        if self.cfg.ppo.fix_encoder:
                            if self.cfg.ppo.iql_q_encoder:
                                # print('======================using iql q encoder======================')
                                self.unio4._policy.obs_encoder.load_state_dict(iql._Q._obs_encoder.state_dict())
                            elif self.cfg.ppo.iql_v_encoder:
                                self.unio4._policy.obs_encoder.load_state_dict(iql._value._obs_encoder.state_dict())
                        if _IQLFT_RESTORE_RNG and total_steps > self.cfg.ppo.online_start_training:
                            _iqlft_restore_rng(rng_snapshot)
                    time2 = time.time()
                    pre_training_time = time.time()
                    pre_training_time = time.time()
                    actor_loss, critic_loss, bc_loss, distill_loss = self.unio4.dp_align_update_no_share(replay_buffer, total_steps)
                    if distill_loss != 0:
                        distill_losses.append(distill_loss)
                    post_training_time = time.time()
                    print('pure policy updated time: {}'.format(post_training_time - pre_training_time))
                    # print('Step: {}, actor_loss: {}, critic_loss: {}'.format(total_steps, actor_loss, critic_loss))
                    time3 = time.time()
                    if self.cfg.training.use_ema and ema is not None:
                        ema.step(self.unio4._policy)
                    ppo_elapsed = getattr(self.unio4, 'last_ppo_elapsed', None)
                    ppo_time_str = f'; ppo loop: {ppo_elapsed:.2f}s' if ppo_elapsed is not None else ''
                    print('step {}; collecting data time: {}; update time: {}{}'.format(total_steps, time2 - time1, time3 - time2, ppo_time_str))
                    replay_buffer.count = 0
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    bc_losses.append(bc_loss)
                    time1 = time.time()
                    if self.cfg.ppo.save_online_cp and update_num % self.cfg.ppo.online_cp_save_freq == 0:
                        self.save_online_checkpoints(online_ft_path, update_num, iql, ema)

                if total_steps % self.cfg.ppo.evaluate_freq == 0:
                    evaluate_num += 1
                    if self.cfg.ppo.idql_eval:
                        idql_log_data = self.unio4_eval(
                                idql_eval = True,
                                dynamics = dynamics,
                                first_action = self.cfg.unio4.first_action,
                                get_np = True,
                                use_gae=self.cfg.unio4.use_gae,
                                iql = iql,
                                Q = Q,
                                repeat_num = 128,
                                eval_times=self.cfg.unio4.eval_times
                                )
                        log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times)
                        all_idql_success_rates.append(idql_log_data['test_mean_score'])
                        all_idql_returns.append(idql_log_data['mean_returns'])
                        if self.cfg.distill_phase == 'online':
                            cm_log_data = self.eval(
                                online=True, eval_times=self.cfg.unio4.eval_times,
                                use_cm=True, distill2mean=self.cfg.distill2mean)
                            cm_all_success_rates.append(cm_log_data['test_mean_score'])
                            cm_all_returns.append(cm_log_data['mean_returns'])
                        else:
                            cm_all_success_rates.append(0)
                            cm_all_returns.append(0)
                    else:
                        log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times)
                        if self.cfg.distill_phase == 'online':
                            cm_log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times, use_cm=True, distill2mean=self.cfg.distill2mean)
                            cm_all_success_rates.append(cm_log_data['test_mean_score'])
                            cm_all_returns.append(cm_log_data['mean_returns'])
                        else:
                            cm_all_success_rates.append(0)
                            cm_all_returns.append(0)
                        all_idql_success_rates.append(0)
                        all_idql_returns.append(0)

                    all_success_rates.append(log_data['test_mean_score'])
                    all_returns.append(log_data['mean_returns'])

                    # Online EMA eval
                    ema_log_data = None
                    if self.cfg.training.use_ema and self.ema_model is not None:
                        ema_log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times,
                                                 policy_override=self.ema_model, eval_name='Online EMA Eval')
                        all_ema_success_rates.append(ema_log_data['test_mean_score'])
                        all_ema_returns.append(ema_log_data['mean_returns'])
                        _, is_updated_ema = self.maybe_update_online_best_ema(ema_log_data['test_mean_score'])
                        if is_updated_ema:
                            print('------------saved online best EMA model----------------')
                    else:
                        all_ema_success_rates.append(0)
                        all_ema_returns.append(0)

                    cprint(
                        'timestep {}: collecting performance: {} evaluate success rates: {}; evaluate returns: {}  actor_loss: {}; critic_loss: {}; bc_loss: {}; distill_loss: {}; cm_SR: {}; cm_ret: {}; idql_SR: {}; idql_ret: {}; ema_SR: {}; ema_ret: {};'.format(
                            total_steps,
                            np.mean(total_episode_r),
                            log_data['test_mean_score'],
                            log_data['mean_returns'],
                            np.mean(actor_losses[int(-self.cfg.ppo.evaluate_freq):]),
                            np.mean(critic_losses[int(-self.cfg.ppo.evaluate_freq):]),
                            np.mean(bc_losses[int(-self.cfg.ppo.evaluate_freq):]),
                            np.mean(distill_losses[int(-self.cfg.ppo.evaluate_freq):]),
                            cm_log_data['test_mean_score'] if self.cfg.distill_phase == 'online' else 0,
                            cm_log_data['mean_returns'] if self.cfg.distill_phase == 'online' else 0,
                            idql_log_data['test_mean_score'] if idql_log_data else 0,
                            idql_log_data['mean_returns'] if idql_log_data else 0,
                            ema_log_data['test_mean_score'] if ema_log_data else 0,
                            ema_log_data['mean_returns'] if ema_log_data else 0,
                        ),
                        'green'
                    )
                    wandb.log({
                        'online ppo success rates': log_data['test_mean_score'],
                        'online ppo returns': log_data['mean_returns'],
                        'online ppo collect returns': np.mean(total_episode_r),
                        'online actor_loss': np.mean(actor_losses[int(-self.cfg.ppo.evaluate_freq):]),
                        'online critic_loss': np.mean(critic_losses[int(-self.cfg.ppo.evaluate_freq):]),
                        'online bc_loss': np.mean(bc_losses[int(-self.cfg.ppo.evaluate_freq):]),
                        'online distill_loss': np.mean(distill_losses[int(-self.cfg.ppo.evaluate_freq):]),
                        'cm_success rates': cm_log_data['test_mean_score'] if self.cfg.distill_phase == 'online' else 0,
                        'cm_returns': cm_log_data['mean_returns'] if self.cfg.distill_phase == 'online' else 0,
                        'idql_success rates': idql_log_data['test_mean_score'] if idql_log_data else 0,
                        'idql_returns': idql_log_data['mean_returns'] if idql_log_data else 0,
                        'online ema success rates': ema_log_data['test_mean_score'] if ema_log_data else 0,
                        'online ema returns': ema_log_data['mean_returns'] if ema_log_data else 0,
                        })
                    # if self.cfg.ppo.iql_ft:
                    #     wandb.log({'Q_loss': np.mean(q_train_losses[int(-self.cfg.ppo.evaluate_freq):]), 'value_loss': np.mean(v_train_losses[int(-self.cfg.ppo.evaluate_freq):])})
                    #     cprint('timestep {} q_loss: {}; v_loss: {}'.format(total_steps, np.mean(q_train_losses[int(-self.cfg.ppo.evaluate_freq):]), np.mean(v_train_losses[int(-self.cfg.ppo.evaluate_freq):])), 'green')

                    os.makedirs(online_ft_path, exist_ok=True)
                    np.savetxt(os.path.join(online_ft_path, 'success_rates.csv'), all_success_rates, fmt='%f', delimiter=',')
                    np.savetxt(os.path.join(online_ft_path, 'returns.csv'), all_returns, fmt='%f', delimiter=',')
                    np.savetxt(os.path.join(online_ft_path, 'idql_success_rates.csv'), all_idql_success_rates, fmt='%f', delimiter=',')
                    np.savetxt(os.path.join(online_ft_path, 'idql_returns.csv'), all_idql_returns, fmt='%f', delimiter=',')
                    np.savetxt(os.path.join(online_ft_path, 'cm_success_rates.csv'), cm_all_success_rates, fmt='%f', delimiter=',')
                    np.savetxt(os.path.join(online_ft_path, 'cm_returns.csv'), cm_all_returns, fmt='%f', delimiter=',')
                    np.savetxt(os.path.join(online_ft_path, 'ema_success_rates.csv'), all_ema_success_rates, fmt='%f', delimiter=',')
                    np.savetxt(os.path.join(online_ft_path, 'ema_returns.csv'), all_ema_returns, fmt='%f', delimiter=',')
        os.makedirs(os.path.join(online_ft_path, 'online_last'), exist_ok=True)

        self.unio4.save(os.path.join(online_ft_path, 'online_last'))
        if self.cfg.training.use_ema and self.ema_model is not None:
            os.makedirs(os.path.join(online_ft_path, 'online_last_ema'), exist_ok=True)
            self.ema_model.save(os.path.join(online_ft_path, 'online_last_ema'))
        self.unio4.flush_ratio_logs(force=True)

    def _online_ft_vec(self, dynamics, Q, iql, iql_online, wandb, online_ft_path, cm_optimizer, cm_lr_scheduler, ema=None, reward_scaler_template=None):
        """Vec env online finetuning branch (ppo.use_vec_env_online=True).
        Uses manual env list (not SubprocVecEnv) to support MultiStepWrapper kwargs."""
        from rl_100.unidpg.online_buffer_vec import ReplayBuffer as VecReplayBuffer
        from rl_100.unidpg.online_buffer import ReplayBuffer as FlatReplayBuffer
        from rl_100.unidpg.uni_ppo import compute_gae_per_env
        import copy as copy_module

        # VIB: optionally force stochastic sampling in online stage while encoder is in eval mode.
        enable_force_stochastic = getattr(self.cfg.ppo, 'force_stochastic_online', True)

        def _set_force_stochastic(encoder, val):
            if hasattr(encoder, 'force_stochastic'):
                encoder.force_stochastic = val

        def _set_iql_deterministic(iql_ref):
            if iql_ref is None:
                return
            encoders = [getattr(iql_ref, 'obs_encoder', None)]
            for net in [iql_ref._Q, iql_ref._target_Q, iql_ref._value]:
                encoders.append(getattr(net, '_obs_encoder', None))
            for encoder in encoders:
                if encoder is not None:
                    encoder.eval()
                    _set_force_stochastic(encoder, False)

        _set_force_stochastic(self.model.obs_encoder, enable_force_stochastic)
        _set_force_stochastic(self.unio4._policy.obs_encoder, enable_force_stochastic)
        _set_iql_deterministic(iql)
        _set_iql_deterministic(iql_online)

        # --- guard unsupported combinations ---
        assert getattr(self.cfg, 'update_phase', 'inloop') != 'outloop', \
            'vec_env v1 does not support update_phase=outloop'
        assert not getattr(self.cfg.ppo, 'iql_adv', False), \
            'vec_env v1 does not support ppo.iql_adv=True'
        assert not getattr(self.cfg.ppo, 'idql_rollout', False), \
            'vec_env v1 does not support ppo.idql_rollout=True'

        train_env_num = getattr(self.cfg.ppo, 'train_env_num', 1)
        env_runner = self.env_runner
        steps_per_update = self.cfg.ppo.batch_size // train_env_num
        assert self.cfg.ppo.batch_size % train_env_num == 0, \
            f'batch_size ({self.cfg.ppo.batch_size}) must be divisible by train_env_num ({train_env_num})'

        use_subproc_vec_rollout = (
            getattr(self.cfg, 'feature_type', None) == '2D'
            and hasattr(env_runner, 'make_subproc_vec_env')
        )
        vec_env = None
        if use_subproc_vec_rollout:
            vec_env = env_runner.make_subproc_vec_env(
                train_env_num,
                record_video_first=False,
                reward_agg_method='discounted_sum',
                gamma=self.cfg.gamma,
            )
            vec_env.seed(int(self.cfg.training.seed))
            envs = None
        else:
            envs = [env_runner.make_env(record_video=False) for _ in range(train_env_num)]
            for env_idx, env in enumerate(envs):
                env.seed(int(self.cfg.training.seed) + env_idx)
        max_steps = self.cfg.task.env_runner.max_steps

        # per-env reward scalers for dynamic scaling
        if self.cfg.ppo.scale_strategy == 'dynamic':
            if reward_scaler_template is None:
                raise RuntimeError(
                    'vec dynamic reward scaling requires a non-null reward_scaler_template')
            import copy as copy_module_std
            reward_scalers = [copy_module_std.deepcopy(reward_scaler_template) for _ in range(train_env_num)]
            for scaler in reward_scalers:
                scaler.reset()

        replay_buffer = VecReplayBuffer(
            args=self.cfg.ppo, shape_info=self.shape_info,
            device=self.device, env_num=train_env_num,
            steps_per_update=steps_per_update)
        replay_buffer.reset()

        iql_ft = getattr(self.cfg.ppo, 'iql_ft', False)
        if iql_ft:
            from rl_100.unidpg.online_buffer import IqlBuffer
            iql_buffer = IqlBuffer(None, args=self.cfg.ppo, shape_info=self.shape_info, device=self.device)

        obs_debug_printed = False

        def stack_obs_dicts(obs_list):
            """Stack vec rollout observations into a batched float tensor dict."""
            nonlocal obs_debug_printed

            if len(obs_list) == 0:
                raise RuntimeError('vec rollout received an empty obs_list')

            expected_keys = tuple(self.shape_info['obs'].keys())
            reference_keys = tuple(obs_list[0].keys())
            missing_from_first = [key for key in expected_keys if key not in reference_keys]
            if missing_from_first:
                raise KeyError(
                    f"vec rollout obs is missing required keys {missing_from_first}; "
                    f"available keys: {sorted(reference_keys)}")

            batched = {}
            for key in expected_keys:
                missing_envs = [idx for idx, obs in enumerate(obs_list) if key not in obs]
                if missing_envs:
                    raise KeyError(
                        f"vec rollout obs key '{key}' missing from env indices {missing_envs}")

                try:
                    stacked = np.stack([obs[key] for obs in obs_list], axis=0)
                except ValueError as exc:
                    shapes = [np.asarray(obs[key]).shape for obs in obs_list]
                    raise ValueError(
                        f"vec rollout obs key '{key}' has inconsistent shapes across envs: {shapes}"
                    ) from exc

                if stacked.size == 0:
                    raise ValueError(f"vec rollout obs key '{key}' produced an empty batch")

                batched[key] = torch.from_numpy(stacked).to(device=self.device, dtype=torch.float)

            if not obs_debug_printed:
                print(f'vec rollout obs keys: {list(batched.keys())}')
                if 'image' in batched:
                    image = batched['image']
                    print(
                        'vec rollout image batch: '
                        f'shape={tuple(image.shape)}, dtype={image.dtype}, '
                        f'min={image.min().item():.4f}, max={image.max().item():.4f}'
                    )
                if 'point_cloud' in batched:
                    point_cloud = batched['point_cloud']
                    print(
                        'vec rollout point_cloud batch: '
                        f'shape={tuple(point_cloud.shape)}, dtype={point_cloud.dtype}'
                    )
                if 'agent_pos' in batched:
                    agent_pos = batched['agent_pos']
                    print(
                        'vec rollout agent_pos batch: '
                        f'shape={tuple(agent_pos.shape)}, dtype={agent_pos.dtype}'
                    )
                obs_debug_printed = True

            return batched

        def unstack_obs_batch(obs_batch_np):
            keys = list(obs_batch_np.keys())
            batch_size = obs_batch_np[keys[0]].shape[0]
            return [
                {k: obs_batch_np[k][i] for k in keys}
                for i in range(batch_size)
            ]

        # --- initial eval ---
        all_success_rates, all_returns = [], []
        cm_all_success_rates, cm_all_returns = [], []
        all_idql_success_rates, all_idql_returns = [], []
        all_ema_success_rates, all_ema_returns = [], []
        if self.cfg.ppo.idql_eval:
            idql_log_data = self.unio4_eval(
                idql_eval=True, dynamics=dynamics,
                first_action=self.cfg.unio4.first_action, get_np=True,
                use_gae=self.cfg.unio4.use_gae, iql=iql, Q=Q,
                repeat_num=128, eval_times=self.cfg.unio4.eval_times)
            all_idql_success_rates.append(idql_log_data['test_mean_score'])
            all_idql_returns.append(idql_log_data['mean_returns'])
            log_data = self.eval(eval_times=self.cfg.unio4.eval_times, online=True)
            if self.cfg.distill_phase == 'online':
                cm_log_data = self.eval(
                    online=True, eval_times=self.cfg.unio4.eval_times,
                    use_cm=True, distill2mean=self.cfg.distill2mean)
                cm_all_success_rates.append(cm_log_data['test_mean_score'])
                cm_all_returns.append(cm_log_data['mean_returns'])
            else:
                cm_all_success_rates.append(0)
                cm_all_returns.append(0)
        else:
            log_data = self.eval(eval_times=self.cfg.unio4.eval_times, online=True)
            if self.cfg.distill_phase == 'online':
                cm_log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times,
                                        use_cm=True, distill2mean=self.cfg.distill2mean)
                cm_all_success_rates.append(cm_log_data['test_mean_score'])
                cm_all_returns.append(cm_log_data['mean_returns'])
            else:
                cm_all_success_rates.append(0)
                cm_all_returns.append(0)
            all_idql_success_rates.append(0)
            all_idql_returns.append(0)
        all_success_rates.append(log_data['test_mean_score'])
        all_returns.append(log_data['mean_returns'])
        # Initial EMA eval
        ema_log_data = None
        if self.cfg.training.use_ema and self.ema_model is not None:
            ema_log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times,
                                     policy_override=self.ema_model, eval_name='Online EMA Eval')
            all_ema_success_rates.append(ema_log_data['test_mean_score'])
            all_ema_returns.append(ema_log_data['mean_returns'])
            _, is_updated_ema = self.maybe_update_online_best_ema(ema_log_data['test_mean_score'])
            if is_updated_ema:
                print('------------saved online best EMA model----------------')
        else:
            all_ema_success_rates.append(0)
            all_ema_returns.append(0)
        cprint('start vec online finetuning, env_num={}, initial policy SR: {}, EMA SR: {}'.format(
            train_env_num, log_data['test_mean_score'],
            ema_log_data['test_mean_score'] if ema_log_data else 'N/A'), 'green')
        wandb.log({
            'online ppo success rates': log_data['test_mean_score'],
            'online ppo returns': log_data['mean_returns'],
            'cm_success rates': cm_all_success_rates[-1] if cm_all_success_rates else 0,
            'cm_returns': cm_all_returns[-1] if cm_all_returns else 0,
            'idql_success rates': all_idql_success_rates[-1] if all_idql_success_rates else 0,
            'idql_returns': all_idql_returns[-1] if all_idql_returns else 0,
            'online ema success rates': ema_log_data['test_mean_score'] if ema_log_data else 0,
            'online ema returns': ema_log_data['mean_returns'] if ema_log_data else 0,
        })

        # --- main loop state ---
        total_steps = 0
        evaluate_num = 0
        next_eval_at = self.cfg.ppo.evaluate_freq
        actor_losses, critic_losses, bc_losses, distill_losses = [], [], [], []
        q_train_losses, v_train_losses = [], []
        total_episode_r = deque(maxlen=10)
        episode_rewards = [0.0] * train_env_num
        episode_steps_per_env = [0] * train_env_num
        update_num = 0
        time1 = time.time()
        idql_log_data = None

        # init per-env obs
        if use_subproc_vec_rollout:
            obs_list = unstack_obs_batch(vec_env.reset())
        else:
            obs_list = [envs[i].reset() for i in range(train_env_num)]

        while total_steps < self.cfg.ppo.max_train_steps:
            if getattr(self.cfg.ppo, 'clip_std_decay', False):
                decay_value = self.value_decay(
                    initial_value=self.cfg.clip_std_max,
                    total_steps=total_steps,
                    max_train_steps=self.cfg.ppo.max_train_steps)
                self.unio4._policy.noise_scheduler.clip_std_max = decay_value

            # save obs before step (for buffer store)
            obs_before_step = [dict(obs) for obs in obs_list]

            # batched policy inference
            obs_dict_input = stack_obs_dicts(obs_list)
            with torch.no_grad():
                action, all_x, a_logprob = self.unio4._policy.all_step_action_logprob(
                    obs_dict_input, fix_encoder=self.cfg.ppo.fix_encoder)

            all_x_np = all_x.detach().cpu().numpy()
            a_logprob_np = a_logprob.detach().cpu().numpy()
            action_np = action.detach().cpu().numpy()

            # per-env step
            next_obs_list = [None] * train_env_num
            step_rewards = np.zeros(train_env_num)
            step_dones = np.zeros(train_env_num)
            step_dws = np.zeros(train_env_num)

            if use_subproc_vec_rollout:
                obs_after_step_np, reward_batch, done_batch, info_batch = vec_env.step(action_np)
                reset_obs_list = unstack_obs_batch(obs_after_step_np)
                for i in range(train_env_num):
                    reward = float(reward_batch[i])
                    done = bool(done_batch[i])
                    info = info_batch[i]

                    episode_rewards[i] += reward
                    episode_steps_per_env[i] += 1

                    dw = done and episode_steps_per_env[i] != max_steps

                    if done and 'terminal_observation' in info:
                        next_obs_list[i] = info['terminal_observation']
                    else:
                        next_obs_list[i] = reset_obs_list[i]

                    if self.cfg.ppo.scale_strategy == 'number':
                        step_rewards[i] = reward * 0.1
                    elif self.cfg.ppo.scale_strategy == 'dynamic':
                        step_rewards[i] = reward_scalers[i](reward)[0]
                    else:
                        step_rewards[i] = reward

                    step_dones[i] = float(done)
                    step_dws[i] = float(dw)

                    if iql_ft:
                        iql_buffer.store(obs=obs_before_step[i], action=all_x_np[-1, i],
                                         reward=reward, next_obs=next_obs_list[i],
                                         done=step_dones[i])

                    if done:
                        total_episode_r.append(episode_rewards[i])
                        print(f'env {i} episode reward: {episode_rewards[i]:.2f}, steps: {episode_steps_per_env[i]}')
                        episode_rewards[i] = 0.0
                        episode_steps_per_env[i] = 0
                        if self.cfg.ppo.scale_strategy == 'dynamic':
                            reward_scalers[i].reset()

                    obs_list[i] = reset_obs_list[i]
            else:
                for i in range(train_env_num):
                    next_obs, reward, done, info = envs[i].step(
                        action_np[i], reward_agg_method='discounted_sum', gamma=self.cfg.gamma)

                    episode_rewards[i] += reward
                    episode_steps_per_env[i] += 1

                    # dw: true termination (not max_steps truncation)
                    dw = done and episode_steps_per_env[i] != max_steps

                    if self.cfg.ppo.scale_strategy == 'number':
                        step_rewards[i] = reward * 0.1
                    elif self.cfg.ppo.scale_strategy == 'dynamic':
                        step_rewards[i] = reward_scalers[i](reward)[0]
                    else:
                        step_rewards[i] = reward

                    step_dones[i] = float(done)
                    step_dws[i] = float(dw)
                    next_obs_list[i] = next_obs  # terminal obs (before reset)

                    # per-env iql buffer store (before auto-reset)
                    # Use raw reward (not scaled) to match offline IQL data distribution
                    if iql_ft:
                        iql_buffer.store(obs=obs_before_step[i], action=all_x_np[-1, i],
                                         reward=reward, next_obs=next_obs_list[i],
                                         done=step_dones[i])

                    # auto-reset
                    if done:
                        total_episode_r.append(episode_rewards[i])
                        print(f'env {i} episode reward: {episode_rewards[i]:.2f}, steps: {episode_steps_per_env[i]}')
                        episode_rewards[i] = 0.0
                        episode_steps_per_env[i] = 0
                        if self.cfg.ppo.scale_strategy == 'dynamic':
                            reward_scalers[i].reset()
                        obs_list[i] = envs[i].reset()
                    else:
                        obs_list[i] = next_obs

            # build batched data for vec buffer
            obs_keys = list(obs_before_step[0].keys())
            obs_batch_np = {k: np.stack([obs_before_step[i][k] for i in range(train_env_num)], axis=0)
                            for k in obs_keys}
            next_obs_batch_np = {k: np.stack([next_obs_list[i][k] for i in range(train_env_num)], axis=0)
                                  for k in obs_keys}

            # all_x: (T+1, train_env_num, ...) -> (train_env_num, T+1, ...)
            all_x_for_buffer = np.moveaxis(all_x_np, 1, 0) if all_x_np.ndim > 2 and all_x_np.shape[1] == train_env_num else all_x_np
            a_logprob_for_buffer = np.moveaxis(a_logprob_np, 1, 0) if a_logprob_np.ndim > 2 and a_logprob_np.shape[1] == train_env_num else a_logprob_np

            replay_buffer.store(obs_batch_np, all_x_for_buffer, a_logprob_for_buffer,
                                step_rewards, next_obs_batch_np, step_dones, step_dws)

            total_steps += train_env_num

            # PPO update when buffer full
            if replay_buffer.count == steps_per_update:
                update_num += 1

                # --- online IQL training (before PPO update) ---
                if iql_ft:
                    if total_steps > self.cfg.ppo.online_start_training:
                        rng_snapshot = _iqlft_snapshot_rng() if _IQLFT_RESTORE_RNG else None
                        print('start online iql training')
                        for _ in range(self.cfg.ppo.iql_steps):
                            alpha = self.cfg.ppo.data_ratio + (1 - self.cfg.ppo.data_ratio) * (total_steps / self.cfg.ppo.max_train_steps)
                            idql_bs = int(getattr(self.cfg.ppo, 'idql_batch_size', 256))
                            online_sample_size = int(alpha * idql_bs)
                            offline_sample_size = idql_bs - online_sample_size
                            online_batch = iql_buffer.sample(batch_size=online_sample_size)
                            offline_batch = self._next_offline_iql_batch_for_online()
                            merged_batch = iql_buffer.merge(online_batch, offline_batch)
                            merged_batch = dict_apply(merged_batch, lambda x: x[:idql_bs])
                            Q_bc_loss, value_loss = iql.update(batch=merged_batch, online=True, pre_cut=True, online_recon=self.cfg.ppo.online_iql_recon)
                        if total_steps >= next_eval_at - self.cfg.ppo.evaluate_freq + train_env_num:
                            print('Step: {}, Q loss: {}, Value loss: {}'.format(total_steps, Q_bc_loss, value_loss))
                            wandb.log({'online iql Q_loss': Q_bc_loss, 'online iql value value_loss': value_loss})
                        q_train_losses.append(Q_bc_loss); v_train_losses.append(value_loss)
                    # encoder backfill
                    if self.cfg.ppo.fix_encoder:
                        if getattr(self.cfg.ppo, 'iql_q_encoder', False):
                            self.unio4._policy.obs_encoder.load_state_dict(iql._Q._obs_encoder.state_dict())
                        elif getattr(self.cfg.ppo, 'iql_v_encoder', False):
                            self.unio4._policy.obs_encoder.load_state_dict(iql._value._obs_encoder.state_dict())
                    if _IQLFT_RESTORE_RNG and total_steps > self.cfg.ppo.online_start_training:
                        _iqlft_restore_rng(rng_snapshot)

                # per-env GAE
                s_vec, a_vec, a_logprob_vec, r_vec, s_vec_, dw_vec, done_vec = \
                    replay_buffer.numpy_to_tensor_vec()

                with torch.no_grad():
                    flat_s = dict_apply(s_vec, lambda x: x.reshape(-1, *x.shape[2:]))
                    flat_s_ = dict_apply(s_vec_, lambda x: x.reshape(-1, *x.shape[2:]))
                    if self.unio4.args.share_encoder:
                        flat_vs, flat_vs_ = self.unio4._compute_critic_values_in_chunks(
                            flat_s, flat_s_, use_obs2latent=True)
                    else:
                        flat_vs, flat_vs_ = self.unio4._compute_critic_values_in_chunks(
                            flat_s, flat_s_, use_obs2latent=False)
                    vs = flat_vs.reshape(steps_per_update, train_env_num, 1)
                    vs_ = flat_vs_.reshape(steps_per_update, train_env_num, 1)

                    adv, v_target = compute_gae_per_env(
                        r_vec, done_vec, dw_vec, vs, vs_,
                        self.cfg.ppo.gamma, self.cfg.ppo.lamda, self.cfg.n_action_steps)

                # create flat buffer for dp_align_update_no_share
                flat_args = copy_module.copy(self.cfg.ppo)
                flat_args.batch_size = steps_per_update * train_env_num
                flat_replay = FlatReplayBuffer(args=flat_args, shape_info=self.shape_info,
                                               device=self.device)

                # flatten vec buffer into flat buffer
                if not replay_buffer.wo_visual:
                    flat_replay.point_cloud = replay_buffer.point_cloud[:steps_per_update].reshape(
                        -1, *replay_buffer.point_cloud.shape[2:])
                    flat_replay.image = replay_buffer.image[:steps_per_update].reshape(
                        -1, *replay_buffer.image.shape[2:])
                    if replay_buffer.use_imagin_robot:
                        flat_replay.imagin_robot = replay_buffer.imagin_robot[:steps_per_update].reshape(
                            -1, *replay_buffer.imagin_robot.shape[2:])
                flat_replay.agent_pos = replay_buffer.agent_pos[:steps_per_update].reshape(
                    -1, *replay_buffer.agent_pos.shape[2:])
                flat_replay.action = replay_buffer.action[:steps_per_update].reshape(
                    -1, *replay_buffer.action.shape[2:])
                flat_replay.a_logprob = replay_buffer.a_logprob[:steps_per_update].reshape(
                    -1, *replay_buffer.a_logprob.shape[2:])
                flat_replay.reward = replay_buffer.reward[:steps_per_update].reshape(-1, 1)
                if not replay_buffer.wo_visual:
                    flat_replay.next_point_cloud = replay_buffer.next_point_cloud[:steps_per_update].reshape(
                        -1, *replay_buffer.next_point_cloud.shape[2:])
                    flat_replay.next_image = replay_buffer.next_image[:steps_per_update].reshape(
                        -1, *replay_buffer.next_image.shape[2:])
                    if replay_buffer.use_imagin_robot:
                        flat_replay.next_imagin_robot = replay_buffer.next_imagin_robot[:steps_per_update].reshape(
                            -1, *replay_buffer.next_imagin_robot.shape[2:])
                flat_replay.next_agent_pos = replay_buffer.next_agent_pos[:steps_per_update].reshape(
                    -1, *replay_buffer.next_agent_pos.shape[2:])
                flat_replay.done = replay_buffer.done[:steps_per_update].reshape(-1, 1)
                flat_replay.dw = replay_buffer.dw[:steps_per_update].reshape(-1, 1)
                flat_replay.count = steps_per_update * train_env_num

                precomputed = {
                    'adv': adv,
                    'v_target': v_target,
                    'vs': flat_vs.reshape(-1, 1),
                }

                time2 = time.time()
                actor_loss, critic_loss, bc_loss, distill_loss = self.unio4.dp_align_update_no_share(
                    flat_replay, total_steps, precomputed=precomputed)
                if distill_loss != 0:
                    distill_losses.append(distill_loss)
                time3 = time.time()
                if self.cfg.training.use_ema and ema is not None:
                    ema.step(self.unio4._policy)
                print(f'step {total_steps}; collecting data time: {time2 - time1:.2f}; '
                      f'update time: {time3 - time2:.2f}')

                replay_buffer.reset()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                bc_losses.append(bc_loss)
                time1 = time.time()

                if getattr(self.cfg.ppo, 'save_online_cp', False) and \
                   update_num % getattr(self.cfg.ppo, 'online_cp_save_freq', 100) == 0:
                    self.save_online_checkpoints(online_ft_path, update_num, iql, ema)

            # eval (threshold-based, not modulo)
            if total_steps >= next_eval_at:
                next_eval_at += self.cfg.ppo.evaluate_freq
                evaluate_num += 1
                if self.cfg.ppo.idql_eval:
                    idql_log_data = self.unio4_eval(
                        idql_eval=True, dynamics=dynamics,
                        first_action=self.cfg.unio4.first_action, get_np=True,
                        use_gae=self.cfg.unio4.use_gae, iql=iql, Q=Q,
                        repeat_num=128, eval_times=self.cfg.unio4.eval_times)
                    log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times)
                    all_idql_success_rates.append(idql_log_data['test_mean_score'])
                    all_idql_returns.append(idql_log_data['mean_returns'])
                    if self.cfg.distill_phase == 'online':
                        cm_log_data = self.eval(
                            online=True, eval_times=self.cfg.unio4.eval_times,
                            use_cm=True, distill2mean=self.cfg.distill2mean)
                        cm_all_success_rates.append(cm_log_data['test_mean_score'])
                        cm_all_returns.append(cm_log_data['mean_returns'])
                    else:
                        cm_all_success_rates.append(0)
                        cm_all_returns.append(0)
                else:
                    log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times)
                    if self.cfg.distill_phase == 'online':
                        cm_log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times,
                                                use_cm=True, distill2mean=self.cfg.distill2mean)
                        cm_all_success_rates.append(cm_log_data['test_mean_score'])
                        cm_all_returns.append(cm_log_data['mean_returns'])
                    else:
                        cm_all_success_rates.append(0)
                        cm_all_returns.append(0)
                    all_idql_success_rates.append(0)
                    all_idql_returns.append(0)

                all_success_rates.append(log_data['test_mean_score'])
                all_returns.append(log_data['mean_returns'])

                # Online EMA eval
                ema_log_data = None
                if self.cfg.training.use_ema and self.ema_model is not None:
                    ema_log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times,
                                             policy_override=self.ema_model, eval_name='Online EMA Eval')
                    all_ema_success_rates.append(ema_log_data['test_mean_score'])
                    all_ema_returns.append(ema_log_data['mean_returns'])
                    _, is_updated_ema = self.maybe_update_online_best_ema(ema_log_data['test_mean_score'])
                    if is_updated_ema:
                        print('------------saved online best EMA model----------------')
                else:
                    all_ema_success_rates.append(0)
                    all_ema_returns.append(0)

                cprint(
                    'timestep {}: collect perf: {} eval SR: {}; eval ret: {} actor_loss: {}; critic_loss: {}; cm_SR: {}; cm_ret: {}; idql_SR: {}; idql_ret: {}; ema_SR: {}; ema_ret: {};'.format(
                        total_steps, np.mean(total_episode_r) if total_episode_r else 0,
                        log_data['test_mean_score'], log_data['mean_returns'],
                        np.mean(actor_losses[-100:]) if actor_losses else 0,
                        np.mean(critic_losses[-100:]) if critic_losses else 0,
                        cm_log_data['test_mean_score'] if self.cfg.distill_phase == 'online' else 0,
                        cm_log_data['mean_returns'] if self.cfg.distill_phase == 'online' else 0,
                        idql_log_data['test_mean_score'] if idql_log_data else 0,
                        idql_log_data['mean_returns'] if idql_log_data else 0,
                        ema_log_data['test_mean_score'] if ema_log_data else 0,
                        ema_log_data['mean_returns'] if ema_log_data else 0,
                    ), 'green')

                wandb.log({
                    'online ppo success rates': log_data['test_mean_score'],
                    'online ppo returns': log_data['mean_returns'],
                    'online ppo collect returns': np.mean(total_episode_r) if total_episode_r else 0,
                    'online actor_loss': np.mean(actor_losses[-100:]) if actor_losses else 0,
                    'online critic_loss': np.mean(critic_losses[-100:]) if critic_losses else 0,
                    'cm_success rates': cm_log_data['test_mean_score'] if self.cfg.distill_phase == 'online' else 0,
                    'cm_returns': cm_log_data['mean_returns'] if self.cfg.distill_phase == 'online' else 0,
                    'idql_success rates': idql_log_data['test_mean_score'] if idql_log_data else 0,
                    'idql_returns': idql_log_data['mean_returns'] if idql_log_data else 0,
                    'online ema success rates': ema_log_data['test_mean_score'] if ema_log_data else 0,
                    'online ema returns': ema_log_data['mean_returns'] if ema_log_data else 0,
                })

                os.makedirs(online_ft_path, exist_ok=True)
                np.savetxt(os.path.join(online_ft_path, 'success_rates.csv'),
                           all_success_rates, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'returns.csv'),
                           all_returns, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'cm_success_rates.csv'),
                           cm_all_success_rates, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'cm_returns.csv'),
                           cm_all_returns, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'idql_success_rates.csv'),
                           all_idql_success_rates, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'idql_returns.csv'),
                           all_idql_returns, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'ema_success_rates.csv'),
                           all_ema_success_rates, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'ema_returns.csv'),
                           all_ema_returns, fmt='%f', delimiter=',')

        # cleanup
        if vec_env is not None:
            vec_env.close()
        else:
            for e in envs:
                e.close() if hasattr(e, 'close') else None
        os.makedirs(os.path.join(online_ft_path, 'online_last'), exist_ok=True)
        self.unio4.save(os.path.join(online_ft_path, 'online_last'))
        if self.cfg.training.use_ema and self.ema_model is not None:
            os.makedirs(os.path.join(online_ft_path, 'online_last_ema'), exist_ok=True)
            self.ema_model.save(os.path.join(online_ft_path, 'online_last_ema'))
        self.unio4.flush_ratio_logs(force=True)

    def unio4_eval(self, idql_eval: bool = False, dynamics = None, first_action = False, get_np = True, use_gae = True, iql = None, Q = None, repeat_num = 100, eval_times: int = 1, use_cm=False, distill2mean=False, eval_name: str = 'IDQL Eval'):
        # Synchronize all processes before evaluation
        if self.is_ddp:
            dist.barrier()

        # Only run evaluation on rank 0 to avoid duplicate evaluation
        if self.rank != 0:
            # Non-rank-0 processes wait here
            if self.is_ddp:
                dist.barrier()  # Wait for rank 0 to finish
            return {'test_mean_score': 0.0, 'mean_returns': 0.0}

        cfg = copy.deepcopy(self.cfg)
        env_runner = self.env_runner
        policy = self.unio4._policy
        if cfg.training.use_ema:
            if cfg.unio4.use_ema_eval:
                policy = self.ema_model

        # VIB: temporarily disable stochastic sampling for deterministic eval.
        saved_force_stochastic = []
        seen_encoders = set()

        def _disable_force_stochastic(encoder):
            if encoder is None or not hasattr(encoder, 'force_stochastic'):
                return
            encoder_id = id(encoder)
            if encoder_id in seen_encoders:
                return
            seen_encoders.add(encoder_id)
            saved_force_stochastic.append((encoder, encoder.force_stochastic))
            encoder.force_stochastic = False

        _disable_force_stochastic(getattr(policy, 'obs_encoder', None))
        if idql_eval and iql is not None:
            _disable_force_stochastic(getattr(iql, 'obs_encoder', None))
            for net_name in ['_Q', '_target_Q', '_value']:
                net = getattr(iql, net_name, None)
                _disable_force_stochastic(getattr(net, '_obs_encoder', None))
        if idql_eval and Q is not None:
            _disable_force_stochastic(getattr(Q, '_obs_encoder', None))
            _disable_force_stochastic(getattr(Q, 'obs_encoder', None))

        policy.eval()
        eval_env_num = getattr(self.cfg.ppo, 'eval_env_num', 1)
        try:
            idql_run_params = inspect.signature(env_runner.idql_run).parameters
        except (TypeError, ValueError):
            idql_run_params = {}
        try:
            run_params = inspect.signature(env_runner.run).parameters
        except (TypeError, ValueError):
            run_params = {}
        log_data = {'test_mean_score': [], 'mean_returns': []}
        try:
            for i in tqdm.tqdm(range(eval_times), desc='evaluating ......'):
                if idql_eval:
                    idql_kwargs = {
                        'dynamics': dynamics,
                        'first_action': first_action,
                        'get_np': get_np,
                        'use_gae': use_gae,
                        'iql': iql,
                        'Q': Q,
                        'repeat_num': repeat_num,
                        'use_cm': use_cm,
                        'distill2mean': distill2mean,
                        'eval_env_num': eval_env_num,
                    }
                    idql_kwargs = {k: v for k, v in idql_kwargs.items() if k in idql_run_params}
                    runner_log = env_runner.idql_run(policy, **idql_kwargs)
                else:
                    run_kwargs = {
                        'use_cm': use_cm,
                        'distill2mean': distill2mean,
                        'eval_env_num': eval_env_num,
                    }
                    run_kwargs = {k: v for k, v in run_kwargs.items() if k in run_params}
                    runner_log = env_runner.run(policy, **run_kwargs)
                cprint(f"---------------- {eval_name} Results --------------", 'magenta')
                for key, value in runner_log.items():
                    if isinstance(value, float):
                        cprint(f"{key}: {value:.4f}", 'magenta')
                log_data['test_mean_score'].append(runner_log['test_mean_score'])
                log_data['mean_returns'].append(runner_log['mean_returns'])
        finally:
            for encoder, prev_value in reversed(saved_force_stochastic):
                encoder.force_stochastic = prev_value
        log_data['test_mean_score'] = np.mean(log_data['test_mean_score'])
        log_data['mean_returns'] = np.mean(log_data['mean_returns'])
        # import pdb; pdb.set_trace()
        print(f'{eval_name} average success rates:', log_data['test_mean_score'])
        print(f'{eval_name} average rewards:', np.mean(log_data['mean_returns']))

        # Synchronize after evaluation
        if self.is_ddp:
            dist.barrier()

        return log_data

    def eval(self, online=False, eval_times=1, use_cm=False, distill2mean=False, policy_override=None, eval_name='Eval'):
        # Synchronize all processes before evaluation
        if self.is_ddp:
            dist.barrier()

        # Only run evaluation on rank 0 to avoid duplicate evaluation
        if self.rank != 0:
            # Non-rank-0 processes wait here
            if self.is_ddp:
                dist.barrier()  # Wait for rank 0 to finish
            return {'test_mean_score': 0.0, 'mean_returns': 0.0}

        # load the latest checkpoint
        # self.output_dir = self.output_dir()
        cfg = copy.deepcopy(self.cfg)

        env_runner = self.env_runner
        if policy_override is not None:
            policy = policy_override
        elif online:
            print('evaluating online policy')
            policy = self.unio4._policy
        else:
            if cfg.training.use_ema:
                policy = self.ema_model
            else:
                policy = self.model_module if self.is_ddp else self.model

        # VIB: temporarily disable stochastic sampling for deterministic eval.
        saved_force_stochastic = []
        if hasattr(policy, 'obs_encoder') and hasattr(policy.obs_encoder, 'force_stochastic'):
            saved_force_stochastic.append((policy.obs_encoder, policy.obs_encoder.force_stochastic))
            policy.obs_encoder.force_stochastic = False
        policy.eval()
        eval_env_num = getattr(self.cfg.ppo, 'eval_env_num', 1)
        # Probe env_runner.run signature so we only forward kwargs it accepts
        # (e.g. pusht/rotate/pour runners don't take eval_env_num). Using
        # signature inspection instead of `except TypeError` avoids silently
        # swallowing real TypeErrors raised inside env_runner.run.
        try:
            run_params = inspect.signature(env_runner.run).parameters
        except (TypeError, ValueError):
            run_params = {}
        base_kwargs = {'use_cm': use_cm, 'distill2mean': distill2mean, 'eval_env_num': eval_env_num}
        run_kwargs = {k: v for k, v in base_kwargs.items() if k in run_params}
        log_data = {'test_mean_score': [], 'mean_returns': []}
        try:
            for _ in range(eval_times):
                runner_log = env_runner.run(policy, **run_kwargs)
                log_data['test_mean_score'].append(runner_log['test_mean_score'])
                log_data['mean_returns'].append(runner_log['mean_returns'])

                if self.rank == 0:
                    cprint(f"---------------- {eval_name} Results --------------", 'magenta')
                    for key, value in runner_log.items():
                        if isinstance(value, float):
                            cprint(f"{key}: {value:.4f}", 'magenta')
        finally:
            for encoder, prev_value in reversed(saved_force_stochastic):
                encoder.force_stochastic = prev_value
        log_data['test_mean_score'] = np.mean(log_data['test_mean_score'])
        log_data['mean_returns'] = np.mean(log_data['mean_returns'])
        print(f'{eval_name} average success rates:', np.mean(log_data['test_mean_score']))
        print(f'{eval_name} average rewards:', np.mean(log_data['mean_returns']))

        # Synchronize after evaluation
        if self.is_ddp:
            dist.barrier()

        return log_data

    def value_decay(self, initial_value, total_steps, max_train_steps, min_value=0.1):
        value_now = initial_value * (1 - total_steps / max_train_steps)
        return np.clip(value_now, a_min=min_value, a_max=None)

    def load_online_checkpoints(self, online_ft_path, iql=None, value_net=None, ema=None):
        self.online_update_num_path = os.path.join(online_ft_path, 'update_num.txt')
        update_num = np.loadtxt(self.online_update_num_path, dtype=int)
        self.online_policy_cp_path = os.path.join(online_ft_path, 'policy', 'update_{}'.format(update_num))
        self.online_value_cp_path = os.path.join(online_ft_path, 'value', 'update_{}'.format(update_num))
        self.online_iql_cp_path = os.path.join(online_ft_path, 'iql', 'update_{}'.format(update_num))
        self.online_lr_cp_path = os.path.join(online_ft_path, 'lr', 'update_{}'.format(update_num), 'lr.txt')
        self.online_distilled_cp_path = os.path.join(online_ft_path, 'distilled', 'update_{}'.format(update_num))
        self.online_update_num = np.loadtxt(self.online_update_num_path, dtype=int)
        self.unio4.load(self.online_policy_cp_path) # 1. load ddim policy
        if hasattr(self.unio4, "critic"):
            self.unio4.load_critic(self.online_value_cp_path) # 2. load value net
        else:
            value_net.load_state_dict(torch.load(os.path.join(self.online_value_cp_path, 'critic.pth'))) #  load value net
            cprint('2. load value net from {}'.format(self.online_value_cp_path), 'green')
        if self.cfg.ppo.iql_ft:
            online_encoder_path = os.path.join(self.online_iql_cp_path, 'encoder.pth')
            if not os.path.exists(online_encoder_path):
                if not self.cfg.ppo.fix_iql_encoder:
                    raise FileNotFoundError(
                        'trainable online IQL encoder checkpoint is required: {}'.format(online_encoder_path)
                    )
                online_encoder_path = None
            iql.load(
            v_path=os.path.join(self.online_iql_cp_path, 'value.pth'),
            q_path=os.path.join(self.online_iql_cp_path, 'Q_bc.pth'),
            encoder_path=online_encoder_path,
            force_load=online_encoder_path is not None
            ) # 3. load iql
            cprint('3. load iql from {}'.format(self.online_iql_cp_path), 'green')
        self.unio4._policy.distilled_model.load_state_dict(torch.load(os.path.join(self.online_distilled_cp_path, 'distilled.pth'))) # 4. load distilled model
        cprint('4. load distilled model from {}'.format(self.online_distilled_cp_path), 'green')
        lr_a, lr_c = np.loadtxt(self.online_lr_cp_path, dtype=float) # 5. load learning rate
        cprint('5. load learning rate from {}'.format(self.online_lr_cp_path), 'green')
        self.cfg.ppo.lr_a, self.cfg.ppo.lr_c = float(lr_a), float(lr_c)
        cprint('load online checkpoint from {}, and actor and critic learning rate is {} and {}'.format(online_ft_path, update_num, self.cfg.ppo.lr_a, self.cfg.ppo.lr_c), 'green')
        # 6. load EMA if available
        if self.cfg.training.use_ema and self.ema_model is not None:
            ema_cp_path = os.path.join(online_ft_path, 'ema', 'update_{}'.format(update_num))
            if os.path.exists(ema_cp_path):
                self.ema_model.load(ema_cp_path)
                cprint('6. load EMA from {}'.format(ema_cp_path), 'green')
                if ema is not None:
                    ema_step_path = os.path.join(ema_cp_path, 'optimization_step.txt')
                    if os.path.exists(ema_step_path):
                        ema.optimization_step = int(np.loadtxt(ema_step_path, dtype=int))
                        cprint('7. load EMA optimization_step from {}'.format(ema_step_path), 'green')
                    else:
                        ema.optimization_step = 0
                        cprint('7. EMA optimization_step not found, reset to 0 for backward compatibility', 'yellow')
            else:
                ema_state = self.ema_model.state_dict()
                policy_state = self.unio4._policy.state_dict()
                filtered_state = {k: v for k, v in policy_state.items() if k in ema_state}
                self.ema_model.load_state_dict(filtered_state, strict=False)
                cprint('6. EMA checkpoint not found, synced from policy', 'yellow')
        if value_net is not None:
            return iql, value_net
        else:
            return iql, self.unio4.critic

    def save_online_checkpoints(self, online_ft_path, update_num, iql=None, ema=None):


        self.online_update_num_path = os.path.join(online_ft_path, 'update_num.txt')
        self.online_policy_cp_path = os.path.join(online_ft_path, 'policy', 'update_{}'.format(update_num))
        self.online_value_cp_path = os.path.join(online_ft_path, 'value', 'update_{}'.format(update_num))
        self.online_iql_cp_path = os.path.join(online_ft_path, 'iql', 'update_{}'.format(update_num))
        self.online_distilled_cp_path = os.path.join(online_ft_path, 'distilled', 'update_{}'.format(update_num))
        self.online_lr_cp_path = os.path.join(online_ft_path, 'lr', 'update_{}'.format(update_num))
        os.makedirs(self.online_policy_cp_path, exist_ok=True)
        os.makedirs(self.online_value_cp_path, exist_ok=True)
        os.makedirs(self.online_iql_cp_path, exist_ok=True)
        os.makedirs(self.online_lr_cp_path, exist_ok=True)
        os.makedirs(self.online_distilled_cp_path, exist_ok=True)
        np.savetxt(self.online_update_num_path, [update_num], fmt='%d', delimiter=',')
        self.unio4.save(self.online_policy_cp_path) # save policy
        self.unio4.save_critic(self.online_value_cp_path) # save value
        if self.cfg.ppo.iql_ft:
            iql.save(
            v_path=os.path.join(self.online_iql_cp_path, 'value.pth'),
            q_path=os.path.join(self.online_iql_cp_path, 'Q_bc.pth'),
            encoder_path=os.path.join(self.online_iql_cp_path, 'encoder.pth')
            ) # save iql
        if self.cfg.distill_phase == 'online':
            torch.save(self.unio4._policy.distilled_model.state_dict(), os.path.join(self.online_distilled_cp_path, 'distilled.pth'))
        np.savetxt(os.path.join(self.online_lr_cp_path, 'lr.txt'), [self.unio4.lr_a, self.unio4.lr_c],fmt='%.10f', delimiter=',') # save learning rate
        if self.cfg.training.use_ema and self.ema_model is not None:
            ema_cp_path = os.path.join(online_ft_path, 'ema', 'update_{}'.format(update_num))
            os.makedirs(ema_cp_path, exist_ok=True)
            self.ema_model.save(ema_cp_path)
            if ema is not None:
                np.savetxt(os.path.join(ema_cp_path, 'optimization_step.txt'),
                           [ema.optimization_step], fmt='%d', delimiter=',')
        print('save online checkpoint to {}'.format(online_ft_path, update_num))

    # @property

    def cleanup_shared_memory(self):
        """Clean up shared memory if this process created it"""
        if hasattr(self, 'shm_manager') and self.shm_manager is not None:
            try:
                print(f"[Rank {self.rank}] Cleaning up shared memory...")
                self.shm_manager.cleanup()

                # Only rank 0 removes the info file
                if self.rank == 0:
                    info_path_file = os.path.join(self.output_dir, 'shared_memory_info_path.txt')
                    if os.path.exists(info_path_file):
                        os.remove(info_path_file)
                        print(f"[Rank 0] Removed shared memory info file")
            except Exception as e:
                print(f"[Rank {self.rank}] Warning: Error during shared memory cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup_shared_memory()


    # @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def sample_batch(self, batch_size: int = 512):
        # all_data = self.all_data
        # data_idxes = torch.from_numpy(np.random.randint(0, all_data['action'].shape[0], size=batch_size))
        # batch = dict_apply(all_data, lambda x: x[data_idxes])

        # For DDP: ensure we're using the dataloader properly
        if not hasattr(self, '_train_iter') or self._train_iter is None:
            # Debug: print dataloader batch size info
            if self.rank == 0:
                print(f"[Debug] DataLoader batch_size: {self.train_dataloader.batch_size}")
                print(f"[Debug] DataLoader total batches: {len(self.train_dataloader)}")
                if hasattr(self.train_dataloader.dataset, '__len__'):
                    print(f"[Debug] Dataset size: {len(self.train_dataloader.dataset)}")
            self._train_iter = iter(self.train_dataloader)

        try:
            batch = next(self._train_iter)
        except StopIteration:
            # Reset iterator when we reach the end
            if self.is_ddp and hasattr(self.train_dataloader, 'sampler'):
                # For DDP, we need to set a new epoch to shuffle data differently
                self._sample_epoch = getattr(self, '_sample_epoch', 0) + 1
                self.train_dataloader.sampler.set_epoch(self._sample_epoch)
            self._train_iter = iter(self.train_dataloader)
            batch = next(self._train_iter)

        return dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

    def sample_finetune_batch(self):
        """Sample a batch from the finetuning dataloader with configurable batch size."""
        # For DDP: ensure we're using the dataloader properly
        if not hasattr(self, '_finetune_iter') or self._finetune_iter is None:
            self._finetune_iter = iter(self.finetune_dataloader)

        try:
            batch = next(self._finetune_iter)
        except StopIteration:
            # Reset iterator when we reach the end
            if self.is_ddp and hasattr(self.finetune_dataloader, 'sampler'):
                # For DDP, we need to set a new epoch to shuffle data differently
                self._finetune_epoch = getattr(self, '_finetune_epoch', 0) + 1
                self.finetune_dataloader.sampler.set_epoch(self._finetune_epoch)
            self._finetune_iter = iter(self.finetune_dataloader)
            batch = next(self._finetune_iter)

        return dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

    def save_checkpoint(self, path=None, tag='latest',
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        }

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    # Special handling for model to save without DDP wrapper
                    if key == 'model' and self.is_ddp and hasattr(value, 'module'):
                        # Save the underlying module state dict (without 'module.' prefix)
                        if use_thread:
                            payload['state_dicts'][key] = _copy_to_cpu(value.module.state_dict())
                        else:
                            payload['state_dicts'][key] = value.module.state_dict()
                    else:
                        if use_thread:
                            payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                        else:
                            payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)

        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())

    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest' or tag=='latest_cm':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best':
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")

    def get_pretrained_model_path(self, stage1_model_name):
        # given a stage1 model name, return the path to the pretrained model
        data_folder_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'third_party',
            'VRL3',
            'src',
            'vrl3data',
        )
        model_folder_path = os.path.join(data_folder_path, "trained_models")
        model_path = os.path.join(model_folder_path, stage1_model_name + '_checkpoint.pth.tar')
        return model_path



    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        def load_state_dict_with_fallback(key, state_dict):
            try:
                self.__dict__[key].load_state_dict(state_dict, **kwargs)
            except (RuntimeError, ValueError) as e:
                if 'optimizer' in key.lower() or 'optim' in key.lower():
                    print(f"Warning: Ignoring state_dict for {key}: {e}")
                    print(f"Skipping optimizer {key} due to state mismatch")
                    return
                raise

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                # Skip model_module if it exists in checkpoint but not in current instance
                if key == 'model_module' and key not in self.__dict__:
                    print(f"Skipping {key} as it doesn't exist in current instance (will be created after DDP wrapping)")
                    continue

                # Handle DDP state dict loading for model
                if key == 'model':
                    # Check if we're loading into a DDP model
                    is_ddp_model = hasattr(self.__dict__[key], 'module')
                    # Check if the saved state dict has 'module.' prefix
                    has_module_prefix = any(k.startswith('module.') for k in value.keys())

                    if is_ddp_model and not has_module_prefix:
                        # Loading non-DDP checkpoint into DDP model - add module prefix
                        new_state_dict = {}
                        for k, v in value.items():
                            new_state_dict[f'module.{k}'] = v
                        load_state_dict_with_fallback(key, new_state_dict)
                    elif not is_ddp_model and has_module_prefix:
                        # Loading DDP checkpoint into non-DDP model - remove module prefix
                        new_state_dict = {}
                        for k, v in value.items():
                            if k.startswith('module.'):
                                new_state_dict[k[7:]] = v
                            else:
                                new_state_dict[k] = v
                        load_state_dict_with_fallback(key, new_state_dict)
                    else:
                        # Keys match - load directly
                        load_state_dict_with_fallback(key, value)
                elif key == 'ema_model' and 'ema_model' in self.__dict__:
                    # Also handle EMA model loading with module prefix
                    has_module_prefix = any(k.startswith('module.') for k in value.keys())
                    if has_module_prefix:
                        new_state_dict = {}
                        for k, v in value.items():
                            if k.startswith('module.'):
                                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
                            else:
                                new_state_dict[k] = v
                        load_state_dict_with_fallback(key, new_state_dict)
                    else:
                        load_state_dict_with_fallback(key, value)
                else:
                    if key in self.__dict__:
                        load_state_dict_with_fallback(key, value)
                    else:
                        print(f"Warning: Skipping {key} as it doesn't exist in current instance")
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])

    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None,
            include_keys=None,
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(cls, path,
            exclude_keys=None,
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)

    def cleanup_shared_memory(self):
        """Clean up shared memory if this process created it"""
        if hasattr(self, 'shm_manager') and self.shm_manager is not None:
            try:
                print(f"[Rank {self.rank}] Cleaning up shared memory...")
                self.shm_manager.cleanup()

                # Only rank 0 removes the info file
                if self.rank == 0:
                    info_path_file = os.path.join(self.output_dir, 'shared_memory_info_path.txt')
                    if os.path.exists(info_path_file):
                        os.remove(info_path_file)
                        print(f"[Rank {self.rank}] Removed shared memory info file")
            except Exception as e:
                print(f"[Rank {self.rank}] Error cleaning up shared memory: {e}")

def setup_ddp():
    """Initialize the distributed environment."""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'rl_100', 'config'))
)
def main(cfg):
    # Setup DDP if running with torchrun
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        setup_ddp()

    workspace = TrainDP3Workspace(cfg)

    try:
        workspace.run()
    except Exception as e:
        print(f"[Rank {workspace.rank if hasattr(workspace, 'rank') else 0}] Training failed with error: {e}")
        raise
    finally:
        # Cleanup shared memory
        workspace.cleanup_shared_memory()

        # Cleanup DDP
        if dist.is_initialized():
            cleanup_ddp()

if __name__ == "__main__":
    main()
