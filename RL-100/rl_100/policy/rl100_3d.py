from typing import Dict
from contextlib import contextmanager
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops
from copy import deepcopy
from rl_100.unidpg.critic import IQL_Q_V_no
from rl_100.model.common.normalizer import LinearNormalizer
from rl_100.policy.base_policy import BasePolicy
from rl_100.model.diffusion.conditional_unet1d import ConditionalUnet1D
from rl_100.model.diffusion.simple_conditional_unet1d import SampleConditionalUnet1D
from rl_100.model.diffusion.mask_generator import LowdimMaskGenerator
from rl_100.common.pytorch_util import dict_apply
from rl_100.common.model_util import print_params
from rl_100.model.vision.pointnet_extractor import DP3Encoder, iDP3Encoder, DP3EncoderReconVIB
from rl_100.model.diffusion.model import MLP, MLPResNet
from rl_100.model.vision.torch_layers import PointNetImaginationExtractorGP
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tqdm import tqdm
import os
from rl_100.model.common.cm_util import update_ema, scalings_for_boundary_conditions, append_dims, extract_into_tensor, predicted_origin, DDIMSolver
class RL1003D(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            cm_noise_scheduler: DDPMScheduler, # DDIM or CM scheduler
            ddim_noise_scheduler: DDPMScheduler,
            scheduler_type: str,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            idp3_pointcloud_encoder_cfg=None,
            joint_opt_encoder=False,
            backbone=None,  
            integrate_strategy="concat",
            model='dp3', # dp3, sample dp3, mlp
            mlp_policy_depth=2,
            action_norm=True,
            use_agent_pos=True,
            act='mish',
            point_downsample=False,
            encoder_type='dp3',
            extractor_name="smallpn",
            use_cm=False,
            ddim_inference_steps=10,
            cm_inference_steps=1,
            policy_layer_norm=True,
            beta_kl: float = 1e-3,
            beta_recon: float = 0.5,
            use_vib: bool = False,
            use_recon: bool = False,
            latent_dim: int = None,
            latent_state_dim: int = None,
            eta: float = 1.0,
            chunk_as_single_action: bool = False,
            # flow matching parameters
            flow_noise_scheduler=None,
            flow_inference_steps: int = 10,
            flow_sde_type: str = 'cps',
            flow_noise_level: float = 0.7,
            flow_sde_window_size: int = 0,
            flow_sigma_safe_max: float = 0.9,
            flow_logit_normal_sampling: bool = False,
            flow_noise_on_final_step: bool = False,
            flow_cps_logprob_mode: str = 'gaussian',
            flow_distill_inference_steps: int = 1,
            flow_distill_teacher_steps: int = 10,
            # parameters passed to step
            **kwargs):
        super().__init__()
        self.use_cm = use_cm
        self.condition_type = condition_type
        self.joint_opt_encoder = joint_opt_encoder
        self.no_pre_action = True
        self.action_norm = action_norm
        self.eta = eta
        self.chunk_as_single_action = chunk_as_single_action
        self.is_flow = (scheduler_type == 'flow')
        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        self.action_dim = action_dim
        if encoder_type == 'idp3':
            pointnet_type = "multi_stage_pointnet"
            obs_encoder = iDP3Encoder(observation_space=obs_dict,
                                                    pointcloud_encoder_cfg=idp3_pointcloud_encoder_cfg,
                                                    use_pc_color=use_pc_color,
                                                    pointnet_type=pointnet_type,
                                                    point_downsample=point_downsample,
                                                    )
        elif encoder_type == 'dp3':
            obs_encoder = DP3Encoder(observation_space=obs_dict,
                                                    img_crop_shape=crop_shape,
                                                    out_channel=encoder_output_dim,
                                                    pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                    use_pc_color=use_pc_color,
                                                    pointnet_type=pointnet_type,
                                                    backbone=backbone,
                                                    integrate_strategy=integrate_strategy,
                                                    use_agent_pos=use_agent_pos,
                                                    )
        elif encoder_type == 'dp3vib':
            obs_encoder = DP3EncoderReconVIB(observation_space=obs_dict,
                                                    img_crop_shape=crop_shape,
                                                    out_channel=encoder_output_dim,
                                                    pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                    use_pc_color=use_pc_color,
                                                    pointnet_type=pointnet_type,
                                                    backbone=backbone,
                                                    integrate_strategy=integrate_strategy,
                                                    use_agent_pos=use_agent_pos,
                                                    beta_kl=beta_kl,
                                                    beta_recon=beta_recon,
                                                    use_vib=use_vib,
                                                    use_recon=use_recon,
                                                    latent_dim=latent_dim,
                                                    latent_state_dim=latent_state_dim,
                                                    )
        elif encoder_type == 'pointnet':
            obs_encoder = PointNetImaginationExtractorGP(
                observation_space=obs_dict,
                out_channel=encoder_output_dim,
                use_agent_pos=use_agent_pos,
                extractor_name=extractor_name,
            )
        else:
            raise ValueError(f"Unsupported encoder type {encoder_type}")
        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim

        self.obs_feature_dim = obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        self.global_cond_dim = global_cond_dim
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        if model == 'sample_dp3':
            print("Sample DP3 used for the denoising model")
            model = SampleConditionalUnet1D(
                input_dim=input_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                condition_type=condition_type,
                use_down_condition=use_down_condition,
                use_mid_condition=use_mid_condition,
                use_up_condition=use_up_condition,
            )
        elif model == 'dp3':
            model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
            )

        elif model == 'mlp':
            model = MLP(
                state_dim=global_cond_dim,
                action_dim=action_dim,
                hidden_dim=diffusion_step_embed_dim,
                depth=mlp_policy_depth,
                device='cuda',
                t_dim=16,
                act=act,
                use_layer_norm=policy_layer_norm,
            )
        elif model == 'skipnet':
            model = MLPResNet(
                state_dim=global_cond_dim,
                hidden_dim=diffusion_step_embed_dim,
                depth=mlp_policy_depth,
                action_dim=action_dim,
                t_dim=16,
                act=act,
                use_layer_norm=policy_layer_norm,
                n_action_steps=n_action_steps,
            )
        self.obs_encoder = obs_encoder
        self.model = model

        # set ddim as ODE solver, distill to consistency model
        self.ddim_scheduler = ddim_noise_scheduler
        self.noise_scheduler = ddim_noise_scheduler
        # cm as pde scheduler
        # if self.use_cm:
        #     self.noise_scheduler = cm_noise_scheduler
        # else:
        #     self.noise_scheduler = ddim_noise_scheduler
        self.cm_scheduler = cm_noise_scheduler

        # Flow matching scheduler setup
        if self.is_flow:
            assert flow_noise_scheduler is not None, "flow_noise_scheduler required when scheduler_type='flow'"
            self.flow_scheduler = flow_noise_scheduler
            self.flow_scheduler.sde_window_size = flow_sde_window_size
            self.flow_scheduler.sigma_safe_max = flow_sigma_safe_max
            self.flow_scheduler.flow_noise_on_final_step = flow_noise_on_final_step
            self.flow_scheduler.cps_logprob_mode = flow_cps_logprob_mode
            self.noise_scheduler = flow_noise_scheduler
            self.ddim_scheduler = flow_noise_scheduler  # reuse for BC forward process
            self.flow_inference_steps = flow_inference_steps
            self.flow_logit_normal_sampling = flow_logit_normal_sampling
            self.flow_distill_inference_steps = flow_distill_inference_steps
            self.flow_distill_teacher_steps = flow_distill_teacher_steps
            self.ddim_inference_steps = flow_inference_steps

        self.ddim_scheduler_pc = copy.deepcopy(self.ddim_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.encoder_type = encoder_type
        self.normalizer = LinearNormalizer()
        self.critic_normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = dict(kwargs)
        for runtime_key in ("use_cm", "distill2mean", "deterministic"):
            self.kwargs.pop(runtime_key, None)

        if not self.is_flow:
            if ddim_inference_steps is None:
                ddim_inference_steps = self.ddim_scheduler.config.num_train_timesteps
            self.ddim_inference_steps = ddim_inference_steps
        self.cm_inference_steps = cm_inference_steps

        if not self.is_flow:
            self.solver = DDIMSolver(
                self.ddim_scheduler.alphas_cumprod.numpy(),
                timesteps=self.ddim_scheduler.config.num_train_timesteps,
                ddim_timesteps=self.ddim_inference_steps,
            )
            self.alpha_schedule = torch.sqrt(self.ddim_scheduler.alphas_cumprod)
            self.sigma_schedule = torch.sqrt(1 - self.ddim_scheduler.alphas_cumprod)

        print_params(self)

        # get action
        if not self.no_pre_action:
            start = 0
        else:
            start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        self.start, self.end = start, end
        if self.is_flow:
            print('===========using dp3 with flow matching policy===========')
        else:
            print('===========using dp3 with ddim to cm polciy===========')
    def get_unet_timesteps(self, timesteps):
        """Convert scheduler timesteps to UNet-compatible integer timesteps.
        For flow matching: parent's timesteps are sigma * N (float), just truncate to int.
        For DDIM/CM: pass through (already integers).
        """
        if self.is_flow:
            if timesteps.is_floating_point():
                return timesteps.long()  # floor: 99.99 → 99, 89.0 → 89
            return timesteps  # already integer
        return timesteps

    def set_target(self):
        # Handle DDP wrapped model - get underlying module for deepcopy
        model_for_copy = self.model.module if hasattr(self.model, 'module') else self.model

        if self.is_flow:
            # Flow distillation: teacher (frozen) + student (trainable), no target_model
            self.teacher = copy.deepcopy(model_for_copy)
            self.teacher.requires_grad_(False)
            self.teacher.load_state_dict(model_for_copy.state_dict())

            self.distilled_model = copy.deepcopy(model_for_copy)

            # Independent scheduler instances
            self.flow_teacher_scheduler = copy.deepcopy(self.flow_scheduler)
            self.flow_student_scheduler = copy.deepcopy(self.flow_scheduler)

            cprint('set teacher and distilled model for flow distillation', 'yellow')
        else:
            self.distilled_model = copy.deepcopy(model_for_copy)
            self.target_model = copy.deepcopy(model_for_copy)
            self.teacher = copy.deepcopy(model_for_copy)
            # freeze target model
            self.target_model.requires_grad_(False)
            self.teacher.requires_grad_(False)
            self.target_model.load_state_dict(model_for_copy.state_dict())
            self.teacher.load_state_dict(model_for_copy.state_dict())
            cprint('set target model, teacher model, and convert to consistency scheduler', 'yellow')

    def promote_distilled_model(self):
        """After flow distillation: swap student weights into self.model and set 1-step inference.
        This makes the distilled student the default policy for all downstream code
        (online rollout, PPO update, eval) without requiring use_cm=True everywhere.
        """
        assert self.is_flow and hasattr(self, 'distilled_model'), \
            "promote_distilled_model requires flow mode with a distilled model"
        model_target = self.model.module if hasattr(self.model, 'module') else self.model
        model_target.load_state_dict(self.distilled_model.state_dict())
        self.flow_inference_steps = self.flow_distill_inference_steps
        self.ddim_inference_steps = self.flow_distill_inference_steps
        cprint(f'promoted distilled student to default model, inference_steps={self.flow_inference_steps}', 'yellow')

    def get_dynamics_encoder(self):
        return self.obs_encoder if self.joint_opt_encoder else deepcopy(self.obs_encoder)

    def initialize_critic(self,
            device,
            q_hidden_dim,
            q_depth,
            q_lr,
            target_update_freq,
            tau,
            gamma,
            v_hidden_dim,
            v_depth,
            v_lr,
            omega,
            is_double_q,
            is_iql,
            is_share_encoder,
            use_action_embed,
            fix_encoder,
            chunk_as_single_action=False,
            n_action_steps=1,
            use_conv_action_embed=False,
            conv_hidden_dims=None,
            conv_latent_cz=32,
            conv_kernel_size=5,
            conv_n_groups=8,
            action_recon_beta=0.5,
            q_layer_norm=False,
            action_embed_layer_norm=False,
            action_scale_norm: bool = False,
            ):
        self.is_iql = is_iql
        if self.joint_opt_encoder:
            critic_obs_encoder = self.obs_encoder
        else:
            critic_obs_encoder = deepcopy(self.obs_encoder)
        if is_iql:
            iql = IQL_Q_V_no(
                device=device, 
                state_dim=self.obs_feature_dim * self.n_obs_steps,
                feature_dim=self.obs_feature_dim, 
                action_dim=self.action_dim, 
                q_hidden_dim=q_hidden_dim, 
                q_depth=q_depth, 
                Q_lr=q_lr,
                target_update_freq=target_update_freq, 
                tau=tau, 
                gamma=gamma, 
                v_hidden_dim=v_hidden_dim, 
                v_depth=v_depth, 
                v_lr=v_lr,
                omega=omega, 
                is_double_q=is_double_q,
                dp3_normalizer=self.critic_normalizer,
                obs_encoder=critic_obs_encoder,
                n_obs_steps=self.n_obs_steps,
                is_share_encoder=is_share_encoder,
                use_pc_color=self.use_pc_color,
                use_action_embed=use_action_embed,
                fix_encoder=fix_encoder,
                chunk_as_single_action=chunk_as_single_action,
                n_action_steps=n_action_steps,
                use_conv_action_embed=use_conv_action_embed,
                conv_hidden_dims=conv_hidden_dims if conv_hidden_dims is not None else [128, 256],
                conv_latent_cz=conv_latent_cz,
                conv_kernel_size=conv_kernel_size,
                conv_n_groups=conv_n_groups,
                action_recon_beta=action_recon_beta,
                q_layer_norm=q_layer_norm,
                action_embed_layer_norm=action_embed_layer_norm,
                action_scale_norm=action_scale_norm,
                )
            Q_bc, value = None, None
        else:
            iql = None
            value = ValueLearner(device, self.obs_feature_dim, v_hidden_dim, v_depth, v_lr, self.critic_normalizer, critic_obs_encoder, self.n_obs_steps,)
            Q_bc = QSarsaLearner(
                                device,
                                self.obs_feature_dim,
                                self.action_dim,
                                q_hidden_dim,
                                q_depth,
                                q_lr,
                                target_update_freq,
                                tau,
                                gamma,
                                self.critic_normalizer,
                                critic_obs_encoder,
                                self.n_obs_steps,
                            )
        return iql, Q_bc, value
    # ========= inference  ============
    def conditional_sample(self,
            condition_data, condition_mask, deterministic=False, use_cm=False, distill2mean=False,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            initial_noise=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):

        if self.is_flow:
            if use_cm and hasattr(self, 'distilled_model'):
                model = self.distilled_model
                scheduler = self.flow_student_scheduler
                num_inference_steps = self.flow_distill_inference_steps
            else:
                model = self.model
                scheduler = self.flow_scheduler
                num_inference_steps = self.flow_inference_steps
        elif use_cm:
            model = self.distilled_model
            scheduler = self.cm_scheduler
            num_inference_steps = self.cm_inference_steps
        else:
            model = self.model
            scheduler = self.ddim_scheduler
            num_inference_steps = self.ddim_inference_steps

        if initial_noise is None:
            trajectory = torch.randn(
                size=condition_data.shape,
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator)
        else:
            if tuple(initial_noise.shape) != tuple(condition_data.shape):
                raise ValueError(
                    "initial_noise shape must match condition_data: "
                    f"{tuple(initial_noise.shape)} != {tuple(condition_data.shape)}"
                )
            trajectory = initial_noise.to(
                device=condition_data.device,
                dtype=condition_data.dtype,
            ).clone()
            if not torch.isfinite(trajectory).all():
                raise ValueError("initial_noise contains NaN or Inf")

        # set step values
        scheduler.set_timesteps(num_inference_steps)


        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # For flow: convert scheduler timestep to UNet integer timestep
            unet_t = self.get_unet_timesteps(t.unsqueeze(0)).squeeze(0) if self.is_flow else t

            model_output = model(sample=trajectory,
                                timestep=unet_t,
                                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            if self.is_flow:
                if deterministic:
                    trajectory = scheduler.step_mean(
                        model_output, t, trajectory).prev_sample
                else:
                    trajectory = scheduler.step(
                        model_output, t, trajectory).prev_sample
            elif use_cm:
                if distill2mean:
                        trajectory = scheduler.step_mean(
                            model_output, t, trajectory, ).denoised
                else:
                    trajectory = scheduler.step(
                        model_output, t, trajectory, eta=self.eta).denoised
            else:
                if deterministic:
                    trajectory = scheduler.step_mean(
                        model_output, t, trajectory, ).prev_sample
                else:
                    trajectory, _ = scheduler.step_logprob(
                        model_output, t, trajectory)
                    trajectory = trajectory.prev_sample


        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory
    def obs2feature(self, obs_dict: Dict[str, torch.Tensor], fix_encoder: bool = False) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        """
        # normalize input
        # import pdb; pdb.set_trace()
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud'] # (batch size, n_obs_steps, number of point, xyz)
        
        # nobs{'point_cloud': tensor torch.Size([1, 2, 512, 3]); 'agent_pos': tensor torch.Size([1, 2, 24])}

        value = next(iter(nobs.values())) # value = point cloud

        B, To = value.shape[:2]# batch size, n obs step
        if self.no_pre_action:
            T = self.horizon - (self.n_obs_steps - 1)
        else:
            T = self.horizon 
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        
        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        was_training = self.obs_encoder.training
        if self.obs_as_global_cond: # usually True
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]).to(self.device)) # [2 i.e. batch_size * n_obs, 512, 3], [2, 24]
            if fix_encoder:
                self.obs_encoder.eval()
                with torch.no_grad():
                    nobs_features = self.obs_encoder(this_nobs)
            else:
                nobs_features = self.obs_encoder(this_nobs) # [2, 128]
            if "cross_attention" in self.condition_type: # False
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1) # i.e. [batch_size, n_obs * state_dim]
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]).to(self.device))
            if fix_encoder:
                self.obs_encoder.eval()
                with torch.no_grad():
                    nobs_features = self.obs_encoder(this_nobs)
            else:
                nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True
        if fix_encoder and was_training:
            self.obs_encoder.train()
        return cond_data, cond_mask, local_cond, global_cond, nobs_features.reshape(B, -1)
    def feature2cond(self, nobs_features: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        nobs_features: feature tensor
        """

        B = batch_size
        if self.no_pre_action:
            T = self.horizon - (self.n_obs_steps - 1)
        else:
            T = self.horizon 
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        
        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond: # usually True
            if "cross_attention" in self.condition_type: # False
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1) 
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1) # i.e. [batch_size, n_obs * state_dim] 
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True
        return cond_data, cond_mask, local_cond, global_cond, nobs_features.reshape(B, -1)


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], deterministic: bool = False, distill2mean: bool = False, use_cm: bool = False, initial_noise=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud'] # (batch size, n_obs_steps, number of point, xyz)
        
        # nobs{'point_cloud': tensor torch.Size([1, 2, 512, 3]); 'agent_pos': tensor torch.Size([1, 2, 24])}

        value = next(iter(nobs.values())) # value = point cloud

        B, To = value.shape[:2]# batch size, n obs step
        if self.no_pre_action:
            T = self.horizon - (self.n_obs_steps - 1)
        else:
            T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        
        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond: # usually True
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]).to(self.device)) # [2 i.e. batch_size * n_obs, 512, 3], [2, 24] 
            nobs_features = self.obs_encoder(this_nobs) # [2, 128]
            if "cross_attention" in self.condition_type: # False
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1) 
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1) # [1 i.e. batch_size * n_obs , 256] 
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]).to(self.device))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            deterministic=deterministic,
            use_cm=use_cm,
            distill2mean=distill2mean,
            initial_noise=initial_noise,
            **self.kwargs) # (batch_size, horizon, act_dim)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da] # just for self.obs_as_global_cond == False, if True == nsample
        if self.action_norm:
            action_pred = self.normalizer['action'].unnormalize(naction_pred)
        else:
            action_pred = torch.clip(naction_pred, -1, 1)

        # get action
        if self.no_pre_action:
            start = 0
        else:
            start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction

        result = {
            'action': action,
            'action_pred': action_pred,
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())


    def set_critic_normalizer(self, normalizer: LinearNormalizer):
        
        self.critic_normalizer.load_state_dict(normalizer.state_dict())
    def obs2latent(self, nobs, eval_policy: bool = False):
        nobs = self.normalizer.normalize(nobs)
        batch_size = nobs['point_cloud'].shape[0]
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]).to(self.device))
        was_training = self.obs_encoder.training
        if eval_policy:
            self.obs_encoder.eval()
            nobs_features = self.obs_encoder(this_nobs)
        else:
            nobs_features = self.obs_encoder(this_nobs)
        if was_training and eval_policy:
            self.obs_encoder.train()
        return nobs_features.reshape(batch_size, -1)
    def obs2latent_recon(self, nobs, eval_policy: bool = False):
        nobs = self.normalizer.normalize(nobs)
        batch_size = nobs['point_cloud'].shape[0]
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]).to(self.device))
        was_training = self.obs_encoder.training
        if eval_policy:
            self.obs_encoder.eval()
            vib_recon_loss, loss_items, nobs_features = self.obs_encoder.Recon_VIB_loss(this_nobs)
        else:
            vib_recon_loss, loss_items, nobs_features = self.obs_encoder.Recon_VIB_loss(this_nobs)
        if was_training and eval_policy:
            self.obs_encoder.train()
        return nobs_features.reshape(batch_size, -1), vib_recon_loss, loss_items
                
    def obs2this_nobs(self, nobs):
        nobs = self.normalizer.normalize(nobs)
        batch_size = nobs['point_cloud'].shape[0]
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]).to(self.device))
        return this_nobs
    def compute_loss(self, batch, fix_encoder=False, online=False):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        if online:
            nactions = batch['action']
        else: 
            if self.action_norm:
                nactions = self.normalizer['action'].normalize(batch['action'])
            else:
                nactions = batch['action']
                nactions = torch.clip(nactions, -1, 1)
        if self.no_pre_action:
            nactions = nactions[:, self.n_obs_steps - 1:]
        # import pdb
        # pdb.set_trace()
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
       
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]).to(self.device))
            if self.encoder_type == 'dp3vib':
                vib_recon_loss, loss_items, nobs_features = self.obs_encoder.Recon_VIB_loss(this_nobs)
            else:
                nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]).to(self.device))
            if self.encoder_type == 'dp3vib':
                vib_recon_loss, loss_items, nobs_features = self.obs_encoder.Recon_VIB_loss(this_nobs)
            else:
                nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)


        bsz = trajectory.shape[0]

        if self.is_flow:
            # Flow matching BC loss: sample discrete timestep indices from the
            # training sigma grid (same grid as parent constructor, with shift).
            N = self.noise_scheduler.config.num_train_timesteps
            if self.flow_logit_normal_sampling:
                # Logit-normal: concentrate sampling on mid-sigma indices
                u = torch.randn(bsz, device=trajectory.device)
                t = torch.sigmoid(u)  # ∈ (0, 1), concentrated around 0.5
                # Grid is descending (index 0 = high sigma). Logit-normal is
                # symmetric around 0.5, so (1-t) and t give the same distribution.
                indices = (t * N).long().clamp(0, N - 1)
            else:
                indices = torch.randint(0, N, (bsz,), device=trajectory.device)
            # Get noisy sample, target, and model timesteps from discrete grid
            noisy_trajectory, target, timesteps = self.noise_scheduler.get_training_noisy_sample(
                trajectory, noise, indices)
        else:
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.ddim_scheduler.config.num_train_timesteps,
                (bsz,), device=trajectory.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_trajectory = self.ddim_scheduler.add_noise(
                trajectory, noise, timesteps)
        


        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        # Predict the noise residual / velocity
        pred = self.model(sample=noisy_trajectory,
                        timestep=timesteps,
                            local_cond=local_cond,
                            global_cond=global_cond)

        if not self.is_flow:
            # DDIM/CM: target depends on prediction_type
            pred_type = self.ddim_scheduler.config.prediction_type
            if pred_type == 'epsilon':
                target = noise
            elif pred_type == 'sample':
                target = trajectory
            elif pred_type == 'v_prediction':
                self.ddim_scheduler.alpha_t = self.ddim_scheduler.alpha_t.to(self.device)
                self.ddim_scheduler.sigma_t = self.ddim_scheduler.sigma_t.to(self.device)
                alpha_t, sigma_t = self.ddim_scheduler.alpha_t[timesteps], self.ddim_scheduler.sigma_t[timesteps]
                alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
                sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
                v_t = alpha_t * noise - sigma_t * trajectory
                target = v_t
            else:
                raise ValueError(f"Unsupported prediction type {pred_type}")
        # else: target already set in flow branch above (velocity = noise - trajectory)
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        if self.encoder_type == 'dp3vib':
            loss += vib_recon_loss
        

        loss_dict = {
                'bc_loss': loss.item(),
                'kl_loss': loss_items['kl_loss'] if self.encoder_type == 'dp3vib' else 0.0,
                'recon_loss': loss_items['recon_loss'] if self.encoder_type == 'dp3vib' else 0.0,
            }

        # print(f"t2-t1: {t2-t1:.3f}")
        # print(f"t3-t2: {t3-t2:.3f}")
        # print(f"t4-t3: {t4-t3:.3f}")
        # print(f"t5-t4: {t5-t4:.3f}")
        # print(f"t6-t5: {t6-t5:.3f}")
        
        return loss, loss_dict
    def compute_ddim2cm_loss(self, batch, distill2mean=False, fix_encoder=False, online=False):
        assert not self.is_flow, "Flow v1 does not support CM distillation (compute_ddim2cm_loss)"
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        if online:
            nactions = batch['action']
        else: 
            if self.action_norm:
                nactions = self.normalizer['action'].normalize(batch['action'])
            else:
                nactions = batch['action']
                nactions = torch.clip(nactions, -1, 1)
            if self.no_pre_action:
                nactions = nactions[:, self.n_obs_steps - 1:]
        # import pdb
        # pdb.set_trace()
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        if online:
            teacher = self.model
        else:
            teacher = self.teacher
        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
    #    self.obs_encoder.requires_grad_(False) # freeze encoder
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]).to(self.device))
            with torch.no_grad():
                nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]).to(self.device))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        latents = trajectory 
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
        topk = self.ddim_scheduler.config.num_train_timesteps // self.ddim_inference_steps
        index = torch.randint(0, self.ddim_inference_steps, (batch_size,), device=self.device).long()
        self.solver.to(self.device)
        start_timesteps = self.solver.ddim_timesteps[index]
        timesteps = start_timesteps - topk
        timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

        # 20.4.4. Get boundary scalings for start_timesteps and (end) timesteps.
        c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
        c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
        c_skip, c_out = scalings_for_boundary_conditions(timesteps)
        c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

        noisy_model_input = self.ddim_scheduler.add_noise(latents, noise, start_timesteps)


        start_timesteps, timesteps = start_timesteps.to(self.device), timesteps.to(self.device)
        noise_pred = self.distilled_model(
            sample=noisy_model_input, 
            timestep=start_timesteps, 
            local_cond=local_cond, 
            global_cond=global_cond)

        pred_x_0 = predicted_origin(
            noise_pred,
            start_timesteps,
            noisy_model_input,
            self.ddim_scheduler.config.prediction_type,
            self.alpha_schedule,
            self.sigma_schedule)
        
        c_skip_start, c_out_start = c_skip_start.to(self.device), c_out_start.to(self.device)
        c_skip, c_out = c_skip.to(self.device), c_out.to(self.device)
        model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0
        with torch.no_grad():
            cond_teacher_output = teacher(
                sample=noisy_model_input, 
                timestep=start_timesteps, 
                local_cond=local_cond, 
                global_cond=global_cond)
            
            cond_pred_x0 = predicted_origin(
                cond_teacher_output,
                start_timesteps,
                noisy_model_input,
                self.ddim_scheduler.config.prediction_type,
                self.alpha_schedule,
                self.sigma_schedule)
            x_prev = self.solver.ddim_step(cond_pred_x0, cond_teacher_output, index)
            
        with torch.no_grad():
            target_noise_pred = self.target_model(
                x_prev.float(),
                timesteps,
                local_cond=local_cond, 
                global_cond=global_cond)
            pred_x_0 = predicted_origin(
                target_noise_pred,
                timesteps,
                x_prev,
                self.ddim_scheduler.config.prediction_type,
                self.alpha_schedule,
                self.sigma_schedule)
            target = c_skip * x_prev + c_out * pred_x_0

        model_pred_mean = model_pred.mean()
        target_mean = target.mean()
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss_dict = {'bc_loss': loss.item()}
        return loss, loss_dict

    def compute_ddim2cm_loss_action(self, batch, distill2mean=False, fix_encoder=False, online=False):
        assert not self.is_flow, "Flow v1 does not support CM distillation (compute_ddim2cm_loss_action)"
        with torch.no_grad():
            teacher_output = self.predict_action(batch['obs'], deterministic=distill2mean, use_cm=False)['action']
        distill_output = self.predict_action(batch['obs'], distill2mean=distill2mean, use_cm=True)['action']
        loss = F.mse_loss(distill_output, teacher_output)
        loss_dict = {
            'bc_loss': loss.item(),
        }
        return loss, loss_dict
    
    def compute_ddim2cm_loss_action_same_noise(self, batch, distill2mean=False, fix_encoder=False, online=False):
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert not self.is_flow, "Flow v1 does not support CM distillation (compute_ddim2cm_loss_action_same_noise)"
        obs_dict = batch['obs']
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud'] # (batch size, n_obs_steps, number of point, xyz)
        
        # nobs{'point_cloud': tensor torch.Size([1, 2, 512, 3]); 'agent_pos': tensor torch.Size([1, 2, 24])}

        value = next(iter(nobs.values())) # value = point cloud

        B, batch_To = value.shape[:2]  # batch size, batch obs time dim
        To = self.n_obs_steps
        if batch_To < To:
            raise ValueError(
                f"batch obs time dim {batch_To} is smaller than n_obs_steps {To}"
            )
        if self.no_pre_action:
            T = self.horizon - (self.n_obs_steps - 1)
        else:
            T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        
        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond: # usually True
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]).to(self.device)) # [2 i.e. batch_size * n_obs, 512, 3], [2, 24]
            nobs_features = self.obs_encoder(this_nobs) # [2, 128]
            if "cross_attention" in self.condition_type: # False
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1) 
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1) # [1 i.e. batch_size * n_obs , 256] 
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]).to(self.device))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        noise_trajectory = torch.randn(
            size=cond_data.shape, 
            dtype=cond_data.dtype,
            device=cond_data.device)

        with torch.no_grad():
            self.ddim_scheduler.set_timesteps(self.ddim_inference_steps)
            trajectory = noise_trajectory
            for i, t in enumerate(self.ddim_scheduler.timesteps):
                # 1. apply conditioning
                trajectory[cond_mask] = cond_data[cond_mask]

                model_output = self.model(sample=trajectory,
                                    timestep=t, 
                                    local_cond=local_cond, global_cond=global_cond)
                
                # if i == len(self.ddim_scheduler.timesteps) - 1:
                #     trajectory = self.ddim_scheduler.step_mean(
                #         model_output, t, trajectory, ).pred_original_sample
                # else:
                if distill2mean:
                    trajectory = self.ddim_scheduler.step_mean(
                        model_output, t, trajectory, ).prev_sample
                else:
                    trajectory = self.ddim_scheduler.step(
                        model_output, t, trajectory).prev_sample

            naction_pred = trajectory[...,:Da] # just for self.obs_as_global_cond == False, if True == nsample
            if self.action_norm:
                action_pred = self.normalizer['action'].unnormalize(naction_pred)
            else:
                action_pred = torch.clip(naction_pred, -1, 1)

            # get action
            if self.no_pre_action:
                start = 0
            else:
                start = To - 1
            end = start + self.n_action_steps
            ddim_action = action_pred[:,start:end]

        #### ----------- defalut cm_inference_steps = 1
        self.cm_scheduler.set_timesteps(self.cm_inference_steps)
        trajectory = noise_trajectory
        for i, t in enumerate(self.cm_scheduler.timesteps):
            # 1. apply conditioning
            trajectory[cond_mask] = cond_data[cond_mask]

            model_output = self.distilled_model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            if i == len(self.cm_scheduler.timesteps) - 1:
                if distill2mean:
                    trajectory = self.cm_scheduler.step_mean(
                        model_output, t, trajectory, ).denoised
                else:
                    trajectory = self.cm_scheduler.step(
                        model_output, t, trajectory).denoised
            else:
                trajectory = self.cm_scheduler.step(
                    model_output, t, trajectory).prev_sample
            
        naction_pred = trajectory[...,:Da] # just for self.obs_as_global_cond == False, if True == nsample
        if self.action_norm:
            action_pred = self.normalizer['action'].unnormalize(naction_pred)
        else:
            action_pred = torch.clip(naction_pred, -1, 1)

        # get action
        if self.no_pre_action:
            start = 0
        else:
            start = To - 1
        end = start + self.n_action_steps
        cm_action = action_pred[:,start:end]

        loss = F.mse_loss(cm_action, ddim_action)
        loss_dict = {
            'bc_loss': loss.item(),
        }
        return loss, loss_dict


    def _extract_action(self, trajectory, Da, To):
        """Extract action from trajectory (shared logic for distillation)."""
        naction_pred = trajectory[..., :Da]
        if self.action_norm:
            action_pred = self.normalizer['action'].unnormalize(naction_pred)
        else:
            action_pred = torch.clip(naction_pred, -1, 1)
        start = 0 if self.no_pre_action else To - 1
        end = start + self.n_action_steps
        return action_pred[:, start:end]

    @contextmanager
    def _deterministic_obs_encoder(self):
        encoder = self.obs_encoder
        was_training = encoder.training
        old_force_stochastic = getattr(encoder, 'force_stochastic', None)
        try:
            encoder.eval()
            if old_force_stochastic is not None:
                encoder.force_stochastic = False
            yield
        finally:
            if old_force_stochastic is not None:
                encoder.force_stochastic = old_force_stochastic
            encoder.train(was_training)

    def compute_flow_distill_loss(self, batch, distill2mean=False, fix_encoder=False):
        """Flow distillation: N-step teacher -> 1-step student, same noise, MSE loss."""
        assert self.is_flow, "compute_flow_distill_loss requires flow mode"

        obs_dict = batch['obs']
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        if self.no_pre_action:
            T = self.horizon - (self.n_obs_steps - 1)
        else:
            T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        # Keep distill conditioning deterministic without changing PPO rollout/update state.
        with torch.no_grad(), self._deterministic_obs_encoder():
            if self.obs_as_global_cond:
                this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]).to(self.device))
                nobs_features = self.obs_encoder(this_nobs)
                if "cross_attention" in self.condition_type:
                    global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
                else:
                    global_cond = nobs_features.reshape(B, -1)
                cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            else:
                this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]).to(self.device))
                nobs_features = self.obs_encoder(this_nobs)
                nobs_features = nobs_features.reshape(B, To, -1)
                cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
                cond_data[:, :To, Da:] = nobs_features
                cond_mask[:, :To, Da:] = True

        noise_trajectory = torch.randn(
            size=cond_data.shape,
            dtype=cond_data.dtype,
            device=cond_data.device)

        # Teacher rollout: N-step flow (use self.model, not frozen self.teacher,
        # so online distill tracks PPO improvements — consistent with DDIM→CM pattern)
        with torch.no_grad():
            self.flow_teacher_scheduler.set_timesteps(self.flow_distill_teacher_steps)
            trajectory = noise_trajectory.clone()
            for t in self.flow_teacher_scheduler.timesteps:
                trajectory[cond_mask] = cond_data[cond_mask]
                unet_t = self.get_unet_timesteps(t.unsqueeze(0)).squeeze(0)
                model_output = self.model(
                    sample=trajectory, timestep=unet_t,
                    local_cond=local_cond, global_cond=global_cond)
                if distill2mean:
                    trajectory = self.flow_teacher_scheduler.step_mean(
                        model_output, t, trajectory).prev_sample
                else:
                    trajectory = self.flow_teacher_scheduler.step(
                        model_output, t, trajectory).prev_sample
            trajectory[cond_mask] = cond_data[cond_mask]
            teacher_action = self._extract_action(trajectory, Da, To)

        # Student rollout: match the teacher branch. Mean distill uses a
        # deterministic target; stochastic distill samples the student step too.
        self.flow_student_scheduler.set_timesteps(self.flow_distill_inference_steps)
        trajectory = noise_trajectory.clone()
        for t in self.flow_student_scheduler.timesteps:
            trajectory[cond_mask] = cond_data[cond_mask]
            unet_t = self.get_unet_timesteps(t.unsqueeze(0)).squeeze(0)
            model_output = self.distilled_model(
                sample=trajectory, timestep=unet_t,
                local_cond=local_cond, global_cond=global_cond)
            if distill2mean:
                trajectory = self.flow_student_scheduler.step_mean(
                    model_output, t, trajectory).prev_sample
            else:
                trajectory = self.flow_student_scheduler.step(
                    model_output, t, trajectory).prev_sample
        trajectory[cond_mask] = cond_data[cond_mask]
        student_action = self._extract_action(trajectory, Da, To)

        loss = F.mse_loss(student_action, teacher_action)
        return loss, {'bc_loss': loss.item()}


    def all_step_logprob(self, state_dict, fix_encoder=True, training=False, use_cm=False):
        cond_data, cond_mask, local_cond, global_cond, nobs_features = self.obs2feature(state_dict, fix_encoder=fix_encoder)

        model = self.model
        if self.is_flow:
            if use_cm and hasattr(self, 'distilled_model'):
                model = self.distilled_model
                scheduler = self.flow_student_scheduler
                num_inference_steps = self.flow_distill_inference_steps
            else:
                scheduler = self.flow_scheduler
                num_inference_steps = self.flow_inference_steps
        elif use_cm:
            scheduler = self.cm_scheduler
            num_inference_steps = self.cm_inference_steps
        else:
            scheduler = self.ddim_scheduler
            num_inference_steps = self.ddim_inference_steps


        batch_size = global_cond.shape[0]

        # init noise x_T
        trajectory = torch.randn(
            size=cond_data.shape,
            dtype=cond_data.dtype,
            device=cond_data.device)

        # set step values
        scheduler.set_timesteps(num_inference_steps)

        all_x, all_next_x, all_logprob = [], [], []
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[cond_mask] = cond_data[cond_mask]
            # 2. time
            timesteps = t
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=self.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(self.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(trajectory.shape[0])

            # For flow: use UNet-compatible timesteps for model call
            unet_timesteps = self.get_unet_timesteps(timesteps)
            model_output = model(sample=trajectory,
                                timestep=unet_timesteps,
                                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            all_x.append(trajectory)
            trajectory, log_prob = scheduler.step_logprob(
                model_output, timesteps, trajectory, )
            trajectory = trajectory.prev_sample
            all_logprob.append(log_prob)

        # finally make sure conditioning is enforced
        trajectory[cond_mask] = cond_data[cond_mask]
        all_x.append(trajectory)
        all_pre_x = all_x[:-1]
        all_next_x = all_x[1:]

        return all_pre_x, all_next_x, all_logprob, cond_data, cond_mask, local_cond, global_cond, nobs_features
    def all_step_action_logprob(self, state_dict, fix_encoder=True, use_cm=False):
        cond_data, cond_mask, local_cond, global_cond, nobs_features = self.obs2feature(state_dict, fix_encoder=fix_encoder)

        model = self.model
        if self.is_flow:
            if use_cm and hasattr(self, 'distilled_model'):
                model = self.distilled_model
                scheduler = self.flow_student_scheduler
                num_inference_steps = self.flow_distill_inference_steps
            else:
                scheduler = self.flow_scheduler
                num_inference_steps = self.flow_inference_steps
        elif use_cm:
            scheduler = self.cm_scheduler
            num_inference_steps = self.cm_inference_steps
        else:
            scheduler = self.ddim_scheduler
            num_inference_steps = self.ddim_inference_steps

        batch_size = global_cond.shape[0]

        # init noise x_T
        trajectory = torch.randn(
            size=cond_data.shape,
            dtype=cond_data.dtype,
            device=cond_data.device)

        # set step values
        scheduler.set_timesteps(num_inference_steps)

        all_x, all_next_x, all_logprob = [], [], []
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[cond_mask] = cond_data[cond_mask]
            # 2. time
            timesteps = t
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=self.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(self.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(trajectory.shape[0])
            unet_timesteps = self.get_unet_timesteps(timesteps)
            model_output = model(sample=trajectory,
                                timestep=unet_timesteps,
                                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            all_x.append(trajectory)
            trajectory, log_prob = scheduler.step_logprob(
                model_output, timesteps, trajectory)
            trajectory = trajectory.prev_sample
            all_logprob.append(log_prob)

        # finally make sure conditioning is enforced
        trajectory[cond_mask] = cond_data[cond_mask]
        all_x.append(trajectory)

        # unnormalize prediction
        naction_pred = trajectory[...,:self.action_dim] # just for self.obs_as_global_cond == False, if True == nsample
        if self.action_norm:
            action_pred = self.normalizer['action'].unnormalize(naction_pred)
        else:
            action_pred = naction_pred

        # get action
        if self.no_pre_action:
            start = 0
        else:
            start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        return action, torch.stack(all_x), torch.stack(all_logprob)
    
    def sample_action_with_logprob(self, state_dict, dynamics=None, first_action=False, use_gae=True, iql=None, Q=None, repeat_num=100, use_cm=False, distill2mean=False):
        """
        结合sample_action的Q值筛选和all_step_logprob的每步记录功能
        
        Args:
            state_dict: observation dictionary
            dynamics: dynamics model for evaluation
            first_action: whether to only use the first action's GAE/Value for action selection
            use_gae: whether to use GAE
            iql: IQL_Q_V object
            Q: Q function
            repeat_num: number of samples for evaluation
            use_cm: whether to use consistency model
            distill2mean: whether to distill to mean
            
        Returns:
            selected_action: 最终选择的action
            selected_all_x: 选中action对应的每步trajectory [num_steps+1, action_dim + obs_dim]
            selected_all_logprob: 选中action对应的每步log probability [num_steps]
            best_idx: 最佳action的索引
        """
        # 扩展state_dict到batch
        state_dict = dict_apply(state_dict, lambda x: x.to(self.device, non_blocking=True))
        state_dict = dict_apply(state_dict, lambda x: torch.repeat_interleave(x, repeats=repeat_num, dim=0))
        
        cond_data, cond_mask, local_cond, global_cond, nobs_features = self.obs2feature(state_dict)

        if self.is_flow:
            if use_cm and hasattr(self, 'distilled_model'):
                model = self.distilled_model
                scheduler = self.flow_student_scheduler
                num_inference_steps = self.flow_distill_inference_steps
            else:
                model = self.model
                scheduler = self.flow_scheduler
                num_inference_steps = self.flow_inference_steps
        elif use_cm:
            model = self.distilled_model
            scheduler = self.cm_scheduler
            num_inference_steps = self.cm_inference_steps
        else:
            model = self.model
            scheduler = self.ddim_scheduler
            num_inference_steps = self.ddim_inference_steps

        batch_size = global_cond.shape[0]
        if batch_size % repeat_num != 0:
            raise ValueError(
                f"Repeated batch size {batch_size} is not divisible by repeat_num={repeat_num}"
            )
        orig_batch_size = batch_size // repeat_num

        # init noise x_T
        trajectory = torch.randn(
            size=cond_data.shape,
            dtype=cond_data.dtype,
            device=cond_data.device)

        # set step values
        scheduler.set_timesteps(num_inference_steps)

        all_x, all_logprob = [], []
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[cond_mask] = cond_data[cond_mask]
            # 2. time
            timesteps = t
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=self.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(self.device)
            # broadcast to batch dimension
            timesteps = timesteps.expand(trajectory.shape[0])
            unet_timesteps = self.get_unet_timesteps(timesteps)

            model_output = model(sample=trajectory,
                                timestep=unet_timesteps,
                                local_cond=local_cond, global_cond=global_cond)

            # 记录当前trajectory
            all_x.append(trajectory.clone())

            # 3. compute previous image: x_t -> x_t-1
            if self.is_flow:
                trajectory, log_prob = scheduler.step_logprob(
                    model_output, timesteps, trajectory)
                trajectory = trajectory.prev_sample
                all_logprob.append(log_prob)
            elif use_cm:
                if distill2mean:
                    trajectory = scheduler.step_mean(
                        model_output, t, trajectory, ).denoised
                else:
                    trajectory = scheduler.step(
                        model_output, t, trajectory, eta=self.eta).denoised
                # 对于CM，我们需要计算log probability（如果scheduler支持）
                if hasattr(scheduler, 'step_logprob'):
                    _, log_prob = scheduler.step_logprob(model_output, timesteps, all_x[-1])
                    all_logprob.append(log_prob)
                else:
                    # 如果不支持，使用零填充或其他合适的值
                    all_logprob.append(torch.zeros(batch_size, device=self.device))
            else:
                trajectory, log_prob = scheduler.step_logprob(
                    model_output, timesteps, trajectory)
                trajectory = trajectory.prev_sample
                all_logprob.append(log_prob)

        # finally make sure conditioning is enforced
        trajectory[cond_mask] = cond_data[cond_mask]
        all_x.append(trajectory.clone())
        
        # 准备用于Q值评估的trajectory
        eval_trajectory = trajectory.clone()
        if not self.no_pre_action:
            eval_trajectory = eval_trajectory[:, self.n_obs_steps - 1:]
            
        # 使用Q值筛选最佳action
        if iql is not None:
            Q = iql.minQ
        
        # dynamics rollout to get Q values
        _, _, _, _, G, gae_advantages = dynamics.multi_step_evaluation(
            nobs_features, eval_trajectory, Q, state_dict=state_dict, use_gae=use_gae)
        
        if use_gae:
            if first_action:
                q_value = gae_advantages[0]
            else:
                if self.n_action_steps > 1:
                    q_value = torch.mean(gae_advantages[: self.n_action_steps], dim=0)
                else:
                    q_value = gae_advantages
        else:
            q_value = G
            
        # 选择最佳action的索引
        q_value_reshaped = q_value.squeeze().reshape(-1, repeat_num)
        best_idx = torch.argmax(q_value_reshaped, dim=1)
        
        # 选择最佳action对应的每步数据
        selected_all_x = []
        selected_all_logprob = []
        
        # 处理all_x
        batch_indices = torch.arange(orig_batch_size, device=self.device)

        for step_x in all_x:
            x_reshaped = step_x.view(orig_batch_size, repeat_num, *step_x.shape[1:])
            selected_x = x_reshaped[batch_indices, best_idx]
            selected_all_x.append(selected_x)
            
        # 处理all_logprob
        for step_logprob in all_logprob:
            logprob_reshaped = step_logprob.view(
                orig_batch_size, repeat_num, *step_logprob.shape[1:]
            )
            selected_logprob = logprob_reshaped[batch_indices, best_idx]
            selected_all_logprob.append(selected_logprob)
        
        # 获取选择的最终action
        final_trajectory = selected_all_x[-1]  # 最后一个trajectory
        
        # unnormalize prediction
        naction_pred = final_trajectory[...,:self.action_dim]
        if self.action_norm:
            action_pred = self.normalizer['action'].unnormalize(naction_pred)
        else:
            action_pred = torch.clip(naction_pred, -1, 1)

        # get action
        if self.no_pre_action:
            start = 0
        else:
            start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        selected_action = action_pred[:,start:end]
        all_logprob = torch.stack(selected_all_logprob)
        return selected_action, torch.stack(selected_all_x), all_logprob
    def sample_action(self, state_dict, dynamics = None, first_action = False, get_np = True, use_gae = True, iql = None, Q = None, repeat_num = 100, batch_size = 256, use_cm=False, distill2mean=False):
        """
        state_dict/nobs_features: must include "obs" key / feature tensor
        dynamics: dynamics model
        first_action: whether to only use the first action's GAE/Value for action selection
        get_np: idql_eval/rollout for OPE
        use_gae: whether to use GAE
        iql: IQL_Q_V object
        Q: Q function
        repeat_num: number of samples for iql_eval
        """
        # state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # if get_np is ture -> inference with iql estimation
        # repeat num times, i.e., batch_size = repeat_num
        # import pdb; pdb.set_trace()
        if isinstance(state_dict, dict):
            state_dict = dict_apply(state_dict, lambda x: x.to(self.device, non_blocking=True))
            if get_np:
                state_dict = dict_apply(state_dict, lambda x: torch.repeat_interleave(x, repeats=repeat_num, dim=0))
            
            cond_data, cond_mask, local_cond, global_cond, nobs_features = self.obs2feature(state_dict, fix_encoder=True)
        elif isinstance(state_dict, torch.Tensor):
            cond_data, cond_mask, local_cond, global_cond, nobs_features = self.feature2cond(state_dict, batch_size=batch_size)
        else:
            raise ValueError(f"Unsupported state_dict type {type(state_dict)}")
        if self.is_flow:
            if use_cm and hasattr(self, 'distilled_model'):
                model = self.distilled_model
                scheduler = self.flow_student_scheduler
                num_inference_steps = self.flow_distill_inference_steps
            else:
                model = self.model
                scheduler = self.flow_scheduler
                num_inference_steps = self.flow_inference_steps
        elif use_cm:
            model = self.distilled_model
            scheduler = self.cm_scheduler
            num_inference_steps = self.cm_inference_steps
        else:
            model = self.model
            scheduler = self.ddim_scheduler
            num_inference_steps = self.ddim_inference_steps
        batch_size = global_cond.shape[0]

        # init noise x_T
        trajectory = torch.randn(
            size=cond_data.shape,
            dtype=cond_data.dtype,
            device=cond_data.device)

        # set step values
        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[cond_mask] = cond_data[cond_mask]
            # 2. time
            timesteps = t
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=self.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(self.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(trajectory.shape[0])
            unet_timesteps = self.get_unet_timesteps(timesteps)
            model_output = model(sample=trajectory,
                                timestep=unet_timesteps,
                                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            if self.is_flow:
                trajectory = scheduler.step(
                    model_output, t, trajectory).prev_sample
            elif use_cm:
                if distill2mean:
                        trajectory = scheduler.step_mean(
                            model_output, t, trajectory, ).denoised
                else:
                    trajectory = scheduler.step(
                        model_output, t, trajectory, eta=self.eta).denoised
            else:
                trajectory = scheduler.step(
                    model_output, t, trajectory, eta=self.eta).prev_sample

        # finally make sure conditioning is enforced
        trajectory[cond_mask] = cond_data[cond_mask] # multi-step actions: batch_size, steps, action_dim
        if not self.no_pre_action:
            trajectory = trajectory[:, self.n_obs_steps - 1:]
        if get_np:
            if iql is not None:
                Q = iql.minQ
            else:
                Q = Q
            # dynamics rollout
            if self.chunk_as_single_action:
                G = dynamics.chunk_evaluation(nobs_features, trajectory, Q, state_dict=state_dict, use_gae=use_gae)
            else:   
                _, _, _, _, G, gae_advantages = dynamics.multi_step_evaluation(nobs_features, trajectory, Q, state_dict=state_dict, use_gae=use_gae)
            # else: # only rollout exacution action steps
            #     all_obs_features, rewards, terminals, infos, G, gae_advantages = self.dynamics.multi_step(nobs_features[: self.n_action_steps], trajectory, Q, use_gae=use_gae)
            if use_gae and not self.chunk_as_single_action:
                if first_action:
                    q_value = gae_advantages[0]
                else:
                    if self.n_action_steps > 1: 
                        q_value = torch.mean(gae_advantages[: self.n_action_steps], dim=0)
                    else:
                        q_value = gae_advantages
            else:
                q_value = G
            orig_batch_size = trajectory.shape[0] // repeat_num
            if trajectory.shape[0] % repeat_num != 0:
                raise ValueError(
                    f"Repeated batch size {trajectory.shape[0]} is not divisible by repeat_num={repeat_num}"
                )
            probs = F.softmax(q_value.reshape(orig_batch_size, repeat_num), dim=1)
            idx = torch.multinomial(probs, 1)
            trajectory = trajectory.reshape(orig_batch_size, repeat_num, *trajectory.shape[1:])
            batch_indices = torch.arange(orig_batch_size, device=trajectory.device)
            naction_pred = trajectory[batch_indices, idx.squeeze(-1)]

            # unnormalize prediction
            naction_pred = naction_pred[...,:naction_pred.shape[-1]] # just for self.obs_as_global_cond == False, if True == nsample
            if self.action_norm:
                action_pred = self.normalizer['action'].unnormalize(naction_pred)
            else:
                action_pred = torch.clip(naction_pred, -1, 1)

            # get action
            if self.no_pre_action:
                start = 0
            else:
                start = self.n_obs_steps - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
            
            # get prediction

            result = {
                'action': action,
                'action_pred': action_pred,
            }
            
            return result
        else:

            return trajectory
    

    
    def train_align(self, replay_buffer, optimizer, fix_encoder, batch_size, encoder_optimizer=None, iterations=10, mini_batch_size=128, log_writer=None):
        
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        batch = replay_buffer.sample(batch_size)
        for step in range(int(iterations)): 
            sub_batch = {}
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, False):
                # Sample replay buffer / batch
                
                sub_batch['action'] = batch['action'][index, -1]
                sub_batch['obs'] = dict_apply(batch['obs'], lambda x: x[index])
                # sample noise that we'll add to the action
                loss, _ = self.compute_loss(sub_batch, fix_encoder, online=True)
                if encoder_optimizer is not None:
                    encoder_optimizer.zero_grad()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                if encoder_optimizer is not None:
                    encoder_optimizer.step()

                metric['actor_loss'].append(0.)
                metric['bc_loss'].append(loss.item())
                metric['ql_loss'].append(0.)
                metric['critic_loss'].append(0.)

        return metric
    def save(
        self, path: str
    ) -> None:
        # Handle DDP wrapped models - get underlying module if wrapped
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        encoder_to_save = self.obs_encoder.module if hasattr(self.obs_encoder, 'module') else self.obs_encoder
        
        torch.save(model_to_save.state_dict(), os.path.join(path, 'model.pt'))
        torch.save(encoder_to_save.state_dict(), os.path.join(path, 'encoder.pt'))
        # if exist distill model
        if hasattr(self, 'distilled_model'):
            # Handle DDP wrapped distilled model
            distilled_to_save = self.distilled_model.module if hasattr(self.distilled_model, 'module') else self.distilled_model
            torch.save(distilled_to_save.state_dict(), os.path.join(path, 'distilled_model.pt'))
        # if exist target model
        if hasattr(self, 'target_model'):
            target_to_save = self.target_model.module if hasattr(self.target_model, 'module') else self.target_model
            torch.save(target_to_save.state_dict(), os.path.join(path, 'target_model.pt'))

        print('Policy parameters saved in {}'.format(path))
    
    def load(
        self, path: str
    ) -> None:
        """Load model weights from path, handling DDP wrapped models."""
        # Handle DDP wrapped models - get underlying module if wrapped
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        encoder_to_load = self.obs_encoder.module if hasattr(self.obs_encoder, 'module') else self.obs_encoder
        
        model_path = os.path.join(path, 'model.pt')
        encoder_path = os.path.join(path, 'encoder.pt')
        
        if os.path.exists(model_path):
            model_to_load.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f'Loaded model from {model_path}')
        if os.path.exists(encoder_path):
            encoder_to_load.load_state_dict(torch.load(encoder_path, map_location='cpu'))
            print(f'Loaded encoder from {encoder_path}')
            
        # Load distilled model if exists
        distilled_path = os.path.join(path, 'distilled_model.pt')
        if os.path.exists(distilled_path) and hasattr(self, 'distilled_model'):
            distilled_to_load = self.distilled_model.module if hasattr(self.distilled_model, 'module') else self.distilled_model
            distilled_to_load.load_state_dict(torch.load(distilled_path, map_location='cpu'))
            print(f'Loaded distilled model from {distilled_path}')
            
        # Load target model if exists
        target_path = os.path.join(path, 'target_model.pt')
        if os.path.exists(target_path) and hasattr(self, 'target_model'):
            target_to_load = self.target_model.module if hasattr(self.target_model, 'module') else self.target_model
            target_to_load.load_state_dict(torch.load(target_path, map_location='cpu'))
            print(f'Loaded target model from {target_path}')

        # Defensive promote: if flow mode and distilled checkpoint exists,
        # ensure default model has student weights and 1-step inference
        if self.is_flow and os.path.exists(distilled_path):
            self.promote_distilled_model()

        print('Policy parameters loaded from {}'.format(path))
