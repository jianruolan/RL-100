import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
from copy import deepcopy

from rl_100.VRL3.src.stage1_models import BasicBlock, ResNet84
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
from .resnet import load_resnet50, load_resnet18
# from .clip import load_clip
from torchvision.ops import FeaturePyramidNetwork
from .model_getter import get_resnet    
import einops
from rl_100.model.diffusion.modules import SinusoidalPosEmb
from rl_100.model.common.modules import SpatialEmb, RandomShiftsAug
from rl_100.common.pytorch_util import dict_apply, replace_submodules
from rl_100.model.vision.crop_randomizer import CropRandomizer
import rl_100.model.vision_3d.point_process as point_process
# from torchmetrics.functional import chamfer_distance
def chamfer_distance(x, y, reduction="mean"):
    """
    x, y:  (B, N, 3) 和 (B, M, 3)
    returns: total_loss, (min_xy, min_yx)
    """
    # 距离矩阵
    dist = torch.cdist(x, y)         # (B, N, M)

    # x -> y
    min_xy, _ = dist.min(dim=-1)     # (B, N)
    # y -> x
    min_yx, _ = dist.min(dim=-2)     # (B, M)

    if reduction == "mean":
        loss = (min_xy.mean(dim=-1) + min_yx.mean(dim=-1)).mean()
    elif reduction == "sum":
        loss = (min_xy.sum(dim=-1) + min_yx.sum(dim=-1)).sum()
    else:            # no reduction
        loss = (min_xy, min_yx)

    return loss, (min_xy, min_yx)    # 跟 PyTorch3D 的格式对齐
def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules

class MLPResNetBlock(nn.Module):
    def __init__(self, features, act, dropout_rate=None, use_layer_norm=False):
        super(MLPResNetBlock, self).__init__()
        self.features = features
        self.act = act
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        
        self.fc1 = nn.Linear(features, features * 4)
        self.fc2 = nn.Linear(features * 4, features)
        self.fc_residual = nn.Linear(features, features) if features != features else None
        
        self.layer_norm = nn.LayerNorm(features) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None and dropout_rate > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        
        if self.fc_residual is not None:
            residual = self.fc_residual(residual)
        
        return residual + x

class PointResNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 hidden_dim: int=256,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 depth: int=2,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.fc_initial = nn.Linear(in_channels, hidden_dim)
        self.blocks = nn.ModuleList([MLPResNetBlock(hidden_dim, act=nn.ReLU(), use_layer_norm=use_layernorm, dropout_rate=0.1) for _ in range(depth)])
        self.activation = nn.ReLU()
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(hidden_dim, out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(hidden_dim, out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):

        x = self.fc_initial(x)
        for block in self.blocks:
            x = block(x)
        x = self.activation(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)

        return x
class PointResNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 hidden_dim: int=256,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 depth: int=2,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.fc_initial = nn.Linear(in_channels, hidden_dim)
        self.blocks = nn.ModuleList([MLPResNetBlock(hidden_dim, act=nn.ReLU(), use_layer_norm=use_layernorm, dropout_rate=0.1) for _ in range(depth)])
        self.activation = nn.ReLU()
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(hidden_dim, out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(hidden_dim, out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):

        x = self.fc_initial(x)
        for block in self.blocks:
            x = block(x)
        x = self.activation(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)

        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()
class Identity(nn.Module):
    def __init__(self, input_placeholder=None):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
from torchvision import datasets, models, transforms
class VRL3Encoder(nn.Module):
    def __init__(self, obs_shape, model_name):
        super().__init__()
        # a wrapper over a non-RL encoder model
        # self.device = device
        assert len(obs_shape) == 3
        self.n_input_channel = obs_shape[0]
        assert self.n_input_channel % 3 == 0
        self.n_images = self.n_input_channel // 3
        self.model = self.init_model(model_name)
        self.model.fc = Identity()
        self.repr_dim = self.model.get_feature_size()

        self.normalize_op = transforms.Normalize((0.485, 0.456, 0.406),
                                                 (0.229, 0.224, 0.225))
        self.channel_mismatch = True

    def init_model(self, model_name):
        # model name is e.g. resnet6_32channel
        n_layer_string, n_channel_string = model_name.split('_')
        layer_string_to_layer_list = {
            'resnet6': [0, 0, 0, 0],
            'resnet10': [1, 1, 1, 1],
            'resnet18': [2, 2, 2, 2],
        }
        channel_string_to_n_channel = {
            '32channel': 32,
            '64channel': 64,
        }
        layer_list = layer_string_to_layer_list[n_layer_string]
        start_num_channel = channel_string_to_n_channel[n_channel_string]
        return ResNet84(BasicBlock, layer_list, start_num_channel=start_num_channel) #.to(self.device)

    def expand_first_layer(self):
        # convolutional channel expansion to deal with input mismatch
        multiplier = self.n_images
        self.model.conv1.weight.data = self.model.conv1.weight.data.repeat(1,multiplier,1,1) / multiplier
        means = (0.485, 0.456, 0.406) * multiplier
        stds = (0.229, 0.224, 0.225) * multiplier
        self.normalize_op = transforms.Normalize(means, stds)
        self.channel_mismatch = False

    def freeze_bn(self):
        # freeze batch norm layers (VRL3 ablation shows modifying how
        # batch norm is trained does not affect performance)
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

    def get_parameters_that_require_grad(self):
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                params.append(param)
        return params

    def transform_obs_tensor_batch(self, obs):
        # transform obs batch before put into the pretrained resnet
        new_obs = self.normalize_op(obs.float()/255)
        return new_obs

    def _forward_impl(self, x):
        x = self.model.get_features(x)
        return x

    def forward(self, obs):
        o = self.transform_obs_tensor_batch(obs)
        h = self._forward_impl(o)
        return h

class MLPResNetBlock(nn.Module):
    def __init__(self, features, act, dropout_rate=None, use_layer_norm=False):
        super(MLPResNetBlock, self).__init__()
        self.features = features
        self.act = act
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        
        self.fc1 = nn.Linear(features, features * 4)
        self.fc2 = nn.Linear(features * 4, features)
        self.fc_residual = nn.Linear(features, features) if features != features else None
        
        self.layer_norm = nn.LayerNorm(features) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None and dropout_rate > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        
        if self.fc_residual is not None:
            residual = self.fc_residual(residual)
        
        return residual + x

class MLPResNetEncoder(nn.Module):
    def __init__(self, state_dim, out_channel, depth=2, dropout_rate=0.1, use_layer_norm=True, hidden_dim=256, act='mish', 
                 t_dim=16,):
        super(MLPResNetEncoder, self).__init__()
        if act == 'mish':
            act = nn.Mish()
        elif act == 'relu':
            act = nn.ReLU()
        else:
            raise NotImplementedError(f"act: {act}, adding more activation functions")
        self.n_output_channels = out_channel
        self.num_blocks = depth
        self.out_dim = out_channel
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activations = act
        input_dim = state_dim
        self.fc_initial = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([MLPResNetBlock(hidden_dim, act=act, use_layer_norm=use_layer_norm, dropout_rate=dropout_rate) for _ in range(depth)])
        self.fc_final = nn.Linear(hidden_dim, out_channel)
        

    def forward(self, obs):
        state = obs['agent_pos']
        x = self.fc_initial(state)
        for block in self.blocks:
            x = block(x)
        x = self.activations(x)
        x = self.fc_final(x)
        return x
    def output_shape(self):
        return self.n_output_channels
class MLPEncoder(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 out_channel=256,
                 depth=2,
                 act='mish', # nn.ReLU(),
                 ):

        super(MLPEncoder, self).__init__()
        if act == 'mish':
            act = nn.Mish()
        elif act == 'relu':
            act = nn.ReLU()
        else:
            raise NotImplementedError(f"act: {act}, adding more activation functions")
        self.n_output_channels = out_channel
        input_dim = state_dim
        layers = [nn.Linear(input_dim, hidden_dim), act]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
        layers.append(nn.Linear(hidden_dim, out_channel))
        layers.append(act)
        self.mid_layer = nn.Sequential(*layers)
        # import pdb; pdb.set_trace()

    def forward(self, obs):
        if isinstance(obs, dict):
            obs = obs['agent_pos']
        embedding = self.mid_layer(obs)
        return embedding
    def output_shape(self):
        return self.n_output_channels

class DrQEncoder(nn.Module):
    def __init__(self, obs_shape, n_channel=32):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = n_channel * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], n_channel, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(n_channel, n_channel, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(n_channel, n_channel, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(n_channel, n_channel, 3, stride=1),
                                     nn.ReLU())

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.contiguous().view(h.shape[0], -1)
        return h
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# class Encoder2D(nn.Module):
#     def __init__(self, obs_shape, out_channel, n_obs_steps, n_channel=32):
#         super().__init__()

#         self.encoder = DrQEncoder(obs_shape, n_channel)
#         self.repr_dim = self.encoder.repr_dim
#         self.rgb_image_key = 'image'
#         self.n_output_channels = out_channel
#         self.n_obs_steps = n_obs_steps
#         self.trunk = nn.Sequential(nn.Linear(self.repr_dim, out_channel),
#                                    nn.LayerNorm(out_channel), nn.Tanh())
#     def forward(self, obs):
#         img = obs[self.rgb_image_key] # batch_size * n_obs_steps, C, H, W
#         # if img.shape[1] != 3:
#         #     img = einops.rearrange(img, "b h w c -> b c h w")
#         img = img.reshape(-1, 3 * self.n_obs_steps, 84, 84) # batch_size, 3 * n_obs_steps, H, W
#         h = self.encoder(img)
#         h = self.trunk(h)
#         return h
#     def output_shape(self):
#         return self.n_output_channels

class DrQEncoder128(nn.Module):
    def __init__(self, obs_shape, n_channel=32):
        super().__init__()

        assert len(obs_shape) == 3  # 确保输入形状为 [C, H, W]
        self.repr_dim = n_channel * 8 * 8  # 输出特征维度

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], n_channel, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(n_channel, n_channel, kernel_size=4, stride=2, padding=1),      # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(n_channel, n_channel, kernel_size=4, stride=2, padding=1),      # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(n_channel, n_channel, kernel_size=4, stride=2, padding=1),      # 16 -> 8
            nn.ReLU()
        )

        self.apply(weight_init)  # 初始化权重

    def forward(self, obs):
        obs = obs / 255.0 - 0.5  # 归一化输入
        h = self.convnet(obs)  # 通过卷积层
        h = h.contiguous().view(h.shape[0], -1)  # 展平特征图
        return h
class Encoder2D(nn.Module):
    def __init__(self, obs_shape, out_channel, n_obs_steps, n_channel=32):
        super().__init__()

        self.encoder = DrQEncoder(obs_shape, n_channel)
        self.repr_dim = self.encoder.repr_dim
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        self.n_obs_steps = n_obs_steps
        self.trunk = nn.Sequential(nn.Linear(self.repr_dim, out_channel),
                                #    nn.LayerNorm(out_channel), 
                                   nn.ReLU())
    def forward(self, obs):
        img = obs[self.rgb_image_key] # batch_size * n_obs_steps, C, H, W
        if img.shape[1] != 3:
            img = einops.rearrange(img, "b h w c -> b c h w")
        # img = img.reshape(-1, 3, 84, 84) # batch_size, 3 * n_obs_steps, H, W
        h = self.encoder(img)
        h = self.trunk(h)
        return h
    def output_shape(self):
        return self.n_output_channels

class DP3Encoder_with2D(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_hidden_dim=256, 
                 state_mlp_activation_fn='mish',
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 feature_type='2D', # 2D, 3D, or 2D3D (encode 2D image feature) or/and 3D (encode 3D point cloud feature) 
                 model_name='resnet6_32channel',
                 use_agent_pos=False,
                 integrate_strategy='concat', # add, concat features of point cloud and image
                 use_pretrained_2DEncoder=False,
                 img_shape=[3, 84, 84],
                 use_visual=True,
                 mlp_depth=2,
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        if use_visual:
            self.n_output_channels = out_channel
        else:
            self.n_output_channels = 0
        self.feature_type = feature_type
        self.use_agent_pos = use_agent_pos
        self.use_visual = use_visual
        self.use_pretrained_2DEncoder = use_pretrained_2DEncoder
        self.img_shape = list(img_shape)
        self.image_normalize = None

        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
        self.integrate_strategy = integrate_strategy
        self.fix_visual_encoder = False

        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        self.model_name = model_name
        # import pdb; pdb.set_trace()
        # use 2d image feature
        if use_visual:
            if feature_type == '2D' or feature_type == '2D3D':
                if use_pretrained_2DEncoder:
                    if model_name == 'resnet_18':
                        self.encoder2D = get_resnet('resnet18', weights='IMAGENET1K_V1')
                        image_channels = int(self.img_shape[0])
                        if image_channels not in (3, 4):
                            raise ValueError(f'ImageNet ResNet18仅支持RGB/RGB-D，当前通道数={image_channels}')
                        if image_channels == 4:
                            old_conv = self.encoder2D.conv1
                            new_conv = nn.Conv2d(
                                4, old_conv.out_channels,
                                kernel_size=old_conv.kernel_size,
                                stride=old_conv.stride,
                                padding=old_conv.padding,
                                bias=False,
                            )
                            with torch.no_grad():
                                new_conv.weight[:, :3].copy_(old_conv.weight)
                                new_conv.weight[:, 3:4].copy_(
                                    old_conv.weight.mean(dim=1, keepdim=True)
                                )
                            self.encoder2D.conv1 = new_conv
                        mean = [0.485, 0.456, 0.406] + ([0.5] if image_channels == 4 else [])
                        std = [0.229, 0.224, 0.225] + ([0.25] if image_channels == 4 else [])
                        self.register_buffer(
                            'image_mean', torch.tensor(mean).view(1, -1, 1, 1),
                            persistent=False,
                        )
                        self.register_buffer(
                            'image_std', torch.tensor(std).view(1, -1, 1, 1),
                            persistent=False,
                        )
                        if out_channel != 512:
                            self.trunk = nn.Sequential(nn.Linear(512, out_channel),
                                                # nn.LayerNorm(out_channel), 
                                                nn.ReLU())
                        else:
                            self.trunk = nn.Identity()
                    elif model_name == 'resnet_50':
                        self.encoder2D, self.image_normalize = load_resnet50(True)
                        self.trunk = nn.Sequential(nn.Linear(2048, out_channel),
                                            # nn.LayerNorm(out_channel), 
                                            nn.ReLU())
                    elif model_name == 'clip':
                        cprint("Load CLIP model", "yellow")
                        self.feature_pyramid = FeaturePyramidNetwork(
                                [64, 256, 512, 1024, 2048], out_channel
                            )
                        self.encoder2D, self.image_normalize = load_clip()
                        # self.trunk = nn.Sequential(nn.Linear(512, out_channel),
                        #                     # nn.LayerNorm(out_channel), 
                        #                     nn.ReLU())
                    else:
                        self.encoder2D = VRL3Encoder([3, 84, 84], model_name)
                        if out_channel != 256:
                            self.trunk = nn.Sequential(nn.Linear(256, out_channel),
                                                # nn.LayerNorm(out_channel), 
                                                nn.ReLU())
                        else:
                            self.trunk = nn.Identity()
                else:
                    if img_shape == [3, 84, 84]:
                        self.encoder2D = DrQEncoder(img_shape, 32)
                    elif img_shape == [3, 128, 128]:
                        self.encoder2D = DrQEncoder128(img_shape, 32)
                    else:
                        raise NotImplementedError(f"img_shape: {img_shape}")
                    if out_channel != self.encoder2D.repr_dim:
                        self.trunk = nn.Sequential(nn.Linear(self.encoder2D.repr_dim, out_channel),
                                            # nn.LayerNorm(out_channel), 
                                            nn.ReLU())
                    else:
                        self.trunk = nn.Identity()
            if self.feature_type == '3D' or self.feature_type == '2D3D':
                self.use_pc_color = use_pc_color
                self.pointnet_type = pointnet_type
                if pointnet_type == "pointnet":
                    if use_pc_color:
                        pointcloud_encoder_cfg.in_channels = 6
                        self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
                    else:
                        pointcloud_encoder_cfg.in_channels = 3
                        self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
                else:
                    raise NotImplementedError(f"pointnet_type: {pointnet_type}")
                cprint(f"[DP3Encoder_with2D] use 3D point cloud and it's shape: {self.point_cloud_shape}", "yellow")
        if self.fix_visual_encoder:
            self.encoder2D.eval()
            for param in self.encoder2D.parameters():
                param.requires_grad = False
            for param in self.trunk.parameters():
                param.requires_grad = False
            cprint("Fix visual encoder", "yellow")
        if use_agent_pos:
            self.n_output_channels  += out_channel
            self.state_mlp = MLPEncoder(
                state_dim=self.state_shape[0], 
                hidden_dim=state_mlp_hidden_dim, 
                out_channel=out_channel, 
                depth=mlp_depth, 
                act=state_mlp_activation_fn,
                )
            cprint(f"[DP3Encoder] use agent_pos information and it's shape: {self.state_shape}", "yellow")
        if feature_type == '2D3D' and self.integrate_strategy == 'concat':
            self.n_output_channels += out_channel
        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")

    def forward(self, observations: Dict) -> torch.Tensor:
        final_feat = None
        if self.use_visual:
            if self.feature_type == '2D' or self.feature_type == '2D3D':
                img = observations[self.rgb_image_key]
                expected_channels = self.img_shape[0]
                if img.shape[1] != expected_channels and img.shape[-1] == expected_channels:
                    img = einops.rearrange(img, "b h w c -> b c h w")
                if img.shape[1] != expected_channels:
                    raise ValueError(
                        f'图像通道数不匹配: expected={expected_channels}, got={tuple(img.shape)}'
                    )
                img = img.float()
                if img.max() >= 10:
                    img = img / 255.0
                if self.model_name == 'resnet_18' and self.use_pretrained_2DEncoder:
                    img = (img - self.image_mean) / self.image_std
                elif self.model_name == 'resnet_50' or self.model_name == 'clip':
                    img = self.image_normalize(img)

                if self.model_name == 'clip':
                    rgb_features = self.encoder2D(img)
                    final_feat = self.feature_pyramid(rgb_features)
                else:
                    final_feat = self.trunk(self.encoder2D(img))
            if self.feature_type == '3D' or self.feature_type == '2D3D':
                points = observations[self.point_cloud_key]
                assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
                if self.use_imagined_robot:
                    img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
                    points = torch.concat([points, img_points], dim=1)
                
                # points = torch.transpose(points, 1, 2)   # B * 3 * N
                # points: B * 3 * (N + sum(Ni))
                pn_feat = self.extractor(points)    # B * out_channel
                if self.feature_type == '2D3D' and self.integrate_strategy == 'add':
                    pn_feat = pn_feat + final_feat
                elif self.feature_type == '2D3D' and self.integrate_strategy == 'concat':
                    final_feat = torch.cat([pn_feat, final_feat], dim=-1)
                else:
                    final_feat = pn_feat
        if self.use_agent_pos:
            state = observations[self.state_key]
            state_feat = self.state_mlp(state)
            if final_feat != None:
                final_feat = torch.cat([final_feat, state_feat], dim=-1)
            else:
                final_feat = state_feat

        return final_feat

    def output_shape(self):
        return self.n_output_channels

    def load_pretrained_encoder(self, model_path, device, verbose=True):
        if self.feature_type == '2D' or self.feature_type == '2D3D':    
            if verbose:

                print("Trying to load pretrained model from:", model_path)
            checkpoint = torch.load(model_path, map_location=torch.device(device))
            state_dict = checkpoint['state_dict']

            pretrained_dict = {}
            # remove `module.` if model was pretrained with distributed mode
            for k, v in state_dict.items():
                if 'module.' in k:
                    name = k[7:]
                else:
                    name = k
                pretrained_dict[name] = v
            self.encoder2D.model.load_state_dict(pretrained_dict, strict=False)
            if verbose:
                cprint("Pretrained model loaded!", "green")
        else:
            cprint("No pretrained model loaded!", "yellow")
    def switch_to_RL_stages(self, verbose=True):
        # run convolutional channel expansion to match input shape
        if self.feature_type == '2D' or self.feature_type == '2D3D':
            self.encoder2D.expand_first_layer()
            if verbose:
                print("Convolutional channel expansion finished: now can take in %d images as input." % self.encoder2D.n_images)


class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 backbone=None,
                 integrate_strategy='add', # add, concat features of point cloud and image
                 use_agent_pos=True,
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        self.use_agent_pos = use_agent_pos
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        # Handle case where point cloud might not be available (RGB-only)
        if self.point_cloud_key in observation_space:
            self.point_cloud_shape = observation_space[self.point_cloud_key]
        else:
            self.point_cloud_shape = None
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
        self.backbone_name = backbone
        self.integrate_strategy = integrate_strategy
        if backbone is not None:   
            assert backbone in ["resnet50", "resnet18", "clip"]
                # Frozen backbone
            if backbone == "resnet50":
                self.backbone, self.image_normalize = load_resnet50()
                self.resnet_fc = nn.Linear(2048, out_channel)
            elif backbone == "resnet18":
                self.backbone, self.image_normalize = load_resnet18()
                self.resnet_fc = nn.Linear(2048, out_channel)
            elif backbone == "clip":
                raise NotImplementedError("CLIP encoder not available. Please install clip: pip install git+https://github.com/openai/CLIP.git")
                # cprint("Load CLIP model", "yellow")
                # self.backbone, self.image_normalize = load_clip()
            for p in self.backbone.parameters():
                p.requires_grad = False
            
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape if self.point_cloud_shape else 'None (RGB-only mode)'}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        
        # Only initialize point cloud extractor if point cloud is available and config is provided
        if self.point_cloud_shape is not None and pointcloud_encoder_cfg is not None:
            if pointnet_type == "pointnet":
                if use_pc_color:
                    pointcloud_encoder_cfg['in_channels'] = 6
                    self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
                else:
                    pointcloud_encoder_cfg['in_channels'] = 3
                    self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
            else:
                raise NotImplementedError(f"pointnet_type: {pointnet_type}")
        else:
            self.extractor = None  # No point cloud extractor for RGB-only case


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]
        
        if use_agent_pos:
            self.n_output_channels  += output_dim
        if backbone is not None and self.integrate_strategy == 'concat':
            self.n_output_channels += out_channel
        
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))
        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        if len(points.shape) != 3:
            print(f"[WARNING] point cloud shape: {points.shape}, expected 3D, got {len(points.shape)}D")
        assert len(points.shape) == 3, f"point cloud shape: {points.shape}, length should be 3"
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        pn_feat = self.extractor(points)    # B * out_channel
        
        # if use 2d image feature
        if self.backbone_name is not None:
            img = observations[self.rgb_image_key]
            # import pdb; pdb.set_trace()
            if img.shape[1] != 3:
                img = einops.rearrange(img, "b h w c -> b c h w")
            img = self.image_normalize(img)
            img_feat = self.backbone(img)
            img_feat = self.resnet_fc(img_feat['res6'])
            if self.integrate_strategy == 'add':
                pn_feat = pn_feat + img_feat
            elif self.integrate_strategy == 'concat':
                pn_feat = torch.cat([pn_feat, img_feat], dim=-1)
            else:
                raise NotImplementedError(f"integrate_strategy: {self.integrate_strategy}")
        if self.use_agent_pos:
            state = observations[self.state_key]
            state_feat = self.state_mlp(state)  # B * 64
            final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        else:
            final_feat = pn_feat
        return final_feat


    def output_shape(self):
        return self.n_output_channels

class DP3EncoderReconVIB(DP3Encoder):
    """DP3 Encoder with optional Variational Information Bottleneck (VIB)
    and optional Reconstruction heads.

    * The underlying DP3Encoder is **not** modified – it still produces
      `final_feat = torch.cat([pn_feat, state_feat], dim=-1)`.
    * Users can toggle:
        - `use_vib`:   whether to learn latent distributions (μ, σ²) and sample.
        - `use_recon`: whether to create decoder heads for point‑cloud/state
                       reconstruction.
      Both flags default to *True* but can be disabled independently.
    * Return values adapt to the active modules to keep调用端简洁。
    """

    def __init__(
        self,
        observation_space: Dict,
        beta_kl: float = 1e-3,
        beta_recon: float = 0.5,
        # ---------------- optional modules ----------------
        use_vib: bool = True,
        use_recon: bool = True,
        # ---------------- VIB bottleneck dims -------------
        latent_dim: int = None,
        latent_state_dim: int = None,
        # ---------------- DP3Encoder kwargs ---------------
        img_crop_shape=None,
        out_channel: int = 256,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color: bool = False,
        pointnet_type: str = "pointnet",
        backbone: str = None,
        integrate_strategy: str = "add",
        use_agent_pos: bool = True,
    ):
        super().__init__(
            observation_space=observation_space,
            img_crop_shape=img_crop_shape,
            out_channel=out_channel,
            state_mlp_size=state_mlp_size,
            state_mlp_activation_fn=state_mlp_activation_fn,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            backbone=backbone,
            integrate_strategy=integrate_strategy,
            use_agent_pos=use_agent_pos,
        )

        self.use_vib = use_vib
        self.use_recon = use_recon
        self.beta_kl = beta_kl
        self.beta_recon = beta_recon
        self.use_agent_pos = use_agent_pos
        self.force_stochastic = False
        # Handle RGB-only case where point cloud might not be available
        if self.point_cloud_key in observation_space:
            num_recon_points = observation_space[self.point_cloud_key][0]
        else:
            num_recon_points = 0  # No point cloud reconstruction for RGB-only case
        state_shape = observation_space['agent_pos']
        state_dim = state_shape[0]
        # feature split dims
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        
        self.out_channel = out_channel
        self.state_feat_dim = state_mlp_size[-1] if use_agent_pos else 0

        # VIB bottleneck dims (None = no reduction, same as input)
        self.latent_pc_dim = latent_dim if latent_dim is not None else out_channel
        self.latent_state_dim = latent_state_dim if latent_state_dim is not None else self.state_feat_dim

        # ------------------- VIB layers -------------------
        if self.use_vib:
            self.latent_mu_pc = nn.Linear(self.out_channel, self.latent_pc_dim)
            self.latent_logvar_pc = nn.Linear(self.out_channel, self.latent_pc_dim)
            if self.use_agent_pos:
                self.latent_mu_state = nn.Linear(self.state_feat_dim, self.latent_state_dim)
                self.latent_logvar_state = nn.Linear(self.state_feat_dim, self.latent_state_dim)
            cprint(
                f"[DP3EncoderVIB] VIB enabled | pc: {out_channel}→{self.latent_pc_dim} | state: {self.state_feat_dim}→{self.latent_state_dim} | beta={beta_kl}",
                "cyan",
            )
        else:
            cprint("[DP3EncoderVIB] VIB disabled", "cyan")

        # ---------------- decoder heads ------------------
        if self.use_recon:
            # Only create point cloud decoder if we have point cloud data
            if num_recon_points > 0:
                self.pc_decoder = nn.Sequential(
                    nn.Linear(self.latent_pc_dim, 512),
                    nn.LayerNorm(512),
                    nn.ReLU(),
                    nn.Linear(512, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Linear(1024, num_recon_points * 3),
                )
            else:
                self.pc_decoder = None  # No point cloud decoder for RGB-only case
            if self.use_agent_pos:
                self.state_decoder = nn.Sequential(
                    nn.Linear(self.latent_state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, state_dim),
                )
            cprint("[DP3EncoderVIB] Reconstruction heads enabled", "cyan")
        else:
            cprint("[DP3EncoderVIB] Reconstruction heads disabled", "cyan")

    def output_shape(self):
        if self.use_vib:
            return self.latent_pc_dim + (self.latent_state_dim if self.use_agent_pos else 0)
        return self.n_output_channels

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def Recon_VIB_loss(self, observations: Dict):
        # 1) get concatenated feature from base encoder
        final_feat = super().forward(observations)  # (B, pc_dim + state_dim)
        pc_feat = final_feat[:, : self.out_channel]
        if self.use_agent_pos:
            state_feat = final_feat[:, self.out_channel :]

        outputs = {}
        # 2) (optional) VIB sampling
        if self.use_vib:
            mu_pc = self.latent_mu_pc(pc_feat)
            logvar_pc = self.latent_logvar_pc(pc_feat)
            if self.training or self.force_stochastic:
                pc_latent = self._reparameterize(mu_pc, logvar_pc)
            else:
                pc_latent = mu_pc
            if self.use_agent_pos:
                # state_feat is not None
                mu_state = self.latent_mu_state(state_feat)
                logvar_state = self.latent_logvar_state(state_feat)
                if self.training or self.force_stochastic:
                    state_latent = self._reparameterize(mu_state, logvar_state)
                else:
                    state_latent = mu_state
                updated_feat = torch.cat([pc_latent, state_latent], dim=-1)
            else:
                mu_state = None
                logvar_state = None
                state_latent = None
                updated_feat = pc_latent
            outputs.update(
                {
                    "pc_latent": pc_latent,
                    "state_latent": state_latent,
                    "mu_pc": mu_pc,
                    "logvar_pc": logvar_pc,
                    "mu_state": mu_state,
                    "logvar_state": logvar_state,
                }
            )
        else:
            # no VIB: use raw features as "latent"
            outputs.update({"pc_latent": pc_feat, "state_latent": state_feat if self.use_agent_pos else None})
            updated_feat = final_feat

        # 3) (optional) Reconstruction
        if self.use_recon:
            recon_pc = self.pc_decoder(outputs["pc_latent"]).reshape(
                pc_feat.size(0), -1, 3
            )
            outputs.update({"recon_pc": recon_pc})
            if outputs.get("state_latent") is not None and hasattr(self, 'state_decoder'):
                recon_state = self.state_decoder(outputs["state_latent"])
                outputs.update({"recon_state": recon_state})

        # compute kl loss if use vib
        loss_items = {}
        if not self.use_vib:
            kl_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        else:
            kl_pc = -0.5 * torch.sum(
                1 + outputs["logvar_pc"] - outputs["mu_pc"].pow(2) - outputs["logvar_pc"].exp(),
                dim=-1,
            ).mean()
            kl_loss = kl_pc
            if outputs.get("mu_state") is not None:
                kl_state = -0.5 * torch.sum(
                    1 + outputs["logvar_state"] - outputs["mu_state"].pow(2) - outputs["logvar_state"].exp(),
                    dim=-1,
                ).mean()
                kl_loss = kl_loss + kl_state
            kl_loss = self.beta_kl * kl_loss
        loss_items.update({"kl_loss": kl_loss.mean().item()})
        # compute reconstruction loss if use reconstruction module 
        if not self.use_recon:
            recon_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        else:
            assert not self.use_imagined_robot, cprint(f"if use imagined robot, please re-compute the ground truth point cloud")
            recon_pc_loss, _ = chamfer_distance(outputs["recon_pc"], observations['point_cloud'])
            recon_loss = recon_pc_loss
            if outputs.get("state_latent") is not None and hasattr(self, 'state_decoder'):
                recon_state_loss = F.mse_loss(outputs["recon_state"], observations['agent_pos'])
                recon_loss = recon_loss + 1.0 * recon_state_loss
            recon_loss = self.beta_recon * recon_loss
        loss_items.update({"recon_loss": recon_loss.mean().item()})
        loss = kl_loss + recon_loss
        return loss, loss_items, updated_feat
    def forward(self, observations: Dict):
        final_feat = super().forward(observations)  # (B, pc_dim + state_dim)
        pc_feat = final_feat[:, : self.out_channel]
        if self.use_agent_pos:
            state_feat = final_feat[:, self.out_channel :]
        # 2) (optional) VIB sampling
        if self.use_vib:
            mu_pc = self.latent_mu_pc(pc_feat)
            logvar_pc = self.latent_logvar_pc(pc_feat)
            if self.training or self.force_stochastic:
                pc_latent = self._reparameterize(mu_pc, logvar_pc)
            else:
                pc_latent = mu_pc
            if self.use_agent_pos:
                # state_feat is not None
                mu_state = self.latent_mu_state(state_feat)
                logvar_state = self.latent_logvar_state(state_feat)
                if self.training or self.force_stochastic:
                    state_latent = self._reparameterize(mu_state, logvar_state)
                else:
                    state_latent = mu_state
                updated_feat = torch.cat([pc_latent, state_latent], dim=-1)
            else:
                updated_feat = pc_latent

        else:
            updated_feat = final_feat
        return updated_feat
    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    @staticmethod
    def _reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class DP3EncoderMultiViewVIB(DP3EncoderReconVIB):
    """DP3 Encoder with Multi-View RGB support and VIB/Reconstruction features.
    
    This class combines:
    1. Multi-view RGB processing from MultiImageObsEncoder
    2. Point cloud processing from DP3Encoder
    3. VIB (Variational Information Bottleneck) from DP3EncoderReconVIB
    4. Reconstruction capabilities from DP3EncoderReconVIB
    
    The encoder processes multiple RGB camera views (e.g., 'image_front', 'image_wrist')
    and combines them with point cloud features before applying VIB and reconstruction.
    """
    
    def __init__(
        self,
        shape_meta: dict,
        observation_space: Dict,
        # Multi-view RGB parameters
        rgb_model: Union[nn.Module, Dict[str, nn.Module]],
        resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        random_crop: bool = True,
        use_group_norm: bool = False,
        share_rgb_model: bool = False,
        imagenet_norm: bool = False,
        # VIB parameters
        beta_kl: float = 5e-4,
        beta_recon: float = 0.5,
        use_vib: bool = True,
        use_recon: bool = True,
        # DP3 parameters
        img_crop_shape=None,
        out_channel: int = 256,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color: bool = False,
        pointnet_type: str = "pointnet",
        integrate_strategy: str = "concat",  # How to combine multi-view and point cloud features
        use_agent_pos: bool = True,
        # Multi-view integration
        rgb_feature_dim: int = 256,  # Dimension of RGB features after encoding
    ):
        # Initialize the parent DP3EncoderReconVIB without backbone
        # We'll handle RGB processing ourselves
        super().__init__(
            observation_space=observation_space,
            beta_kl=beta_kl,
            beta_recon=beta_recon,
            use_vib=use_vib,
            use_recon=use_recon,
            img_crop_shape=img_crop_shape,
            out_channel=out_channel,
            state_mlp_size=state_mlp_size,
            state_mlp_activation_fn=state_mlp_activation_fn,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            backbone=None,  # We handle RGB processing separately
            integrate_strategy="add",  # Parent class integration
            use_agent_pos=use_agent_pos,
        )
        
        # Store multi-view specific parameters
        self.shape_meta = shape_meta
        self.rgb_feature_dim = rgb_feature_dim
        self.multiview_integrate_strategy = integrate_strategy
        
        # Check if point cloud is available in observation space
        self.has_point_cloud = self.point_cloud_key in observation_space
        
        # Initialize multi-view RGB processing components
        rgb_keys = []
        low_dim_keys = []
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()
        
        # Handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model
        
        # Process observation shape metadata
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            
            if type == 'rgb':
                rgb_keys.append(key)
                
                # Configure model for this key
                this_model = None
                if not share_rgb_model:
                    if rgb_model is not None:
                        if isinstance(rgb_model, dict):
                            # Have provided model for each key
                            this_model = rgb_model[key]
                        else:
                            assert isinstance(rgb_model, nn.Module)
                            # Have a copy of the rgb model
                            this_model = copy.deepcopy(rgb_model)
                    else:
                        # Create default ResNet18 model for non-shared mode
                        this_model = get_resnet('resnet18')
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # Configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h, w)
                    )
                    input_shape = (shape[0], h, w)
                
                # Configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(
                            size=(h, w)
                        )
                
                # Configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
                
            elif type == 'low_dim':
                low_dim_keys.append(key)
        
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        
        # Store multi-view components
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        
        # Add projection layer to combine RGB features
        if len(rgb_keys) > 0:
            # Get the actual output dimension from the RGB model
            if share_rgb_model and isinstance(rgb_model, nn.Module):
                # Test with dummy input to get actual output dimension
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 84, 84)
                    actual_rgb_dim = rgb_model(dummy_input).shape[1]
            else:
                # Assume standard ResNet output
                actual_rgb_dim = 512  # Default for ResNet18
            
            if share_rgb_model:
                rgb_out_dim = actual_rgb_dim * len(rgb_keys)
            else:
                rgb_out_dim = actual_rgb_dim * len(rgb_keys)
            
            self.rgb_projection = nn.Sequential(
                nn.Linear(rgb_out_dim, rgb_feature_dim),
                nn.LayerNorm(rgb_feature_dim),
                nn.ReLU()
            )
        
        # Update output channels based on integration strategy
        if self.multiview_integrate_strategy == 'concat' and len(rgb_keys) > 0:
            # Calculate output dimension based on available modalities
            if self.has_point_cloud:
                self.n_output_channels = out_channel + rgb_feature_dim + (self.state_feat_dim if use_agent_pos else 0)
            else:
                # RGB-only case
                self.n_output_channels = rgb_feature_dim + (self.state_feat_dim if use_agent_pos else 0)
            
            # Update VIB layers for concatenated features
            if self.use_vib:
                self.latent_mu_multiview = nn.Linear(rgb_feature_dim, rgb_feature_dim)
                self.latent_logvar_multiview = nn.Linear(rgb_feature_dim, rgb_feature_dim)
        elif not self.has_point_cloud and len(rgb_keys) > 0:
            # RGB-only case with add strategy (doesn't make sense to add without point cloud)
            self.n_output_channels = rgb_feature_dim + (self.state_feat_dim if use_agent_pos else 0)
        
        cprint(f"[DP3EncoderMultiViewVIB] RGB keys: {rgb_keys}", "yellow")
        cprint(f"[DP3EncoderMultiViewVIB] Integration strategy: {self.multiview_integrate_strategy}", "yellow")
        cprint(f"[DP3EncoderMultiViewVIB] Output dimension: {self.n_output_channels}", "cyan")
    
    def process_rgb_observations(self, obs_dict):
        """Process multi-view RGB observations."""
        batch_size = None
        features = []
        
        # Process RGB inputs
        if self.share_rgb_model:
            # Pass all RGB obs to shared RGB model
            imgs = []
            for key in self.rgb_keys:
                img = obs_dict[key]
                if img.shape[1] != 3:
                    img = img.permute(0, 3, 1, 2)  # b h w c -> b c h w
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            
            if len(imgs) > 0:
                # (N*B, C, H, W)
                imgs = torch.cat(imgs, dim=0)
                # (N*B, D)
                feature = self.key_model_map['rgb'](imgs)
                # (N, B, D)
                feature = feature.reshape(-1, batch_size, *feature.shape[1:])
                # (B, N, D)
                feature = torch.moveaxis(feature, 0, 1)
                # (B, N*D)
                feature = feature.reshape(batch_size, -1)
                features.append(feature)
        else:
            # Run each RGB obs through independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if img.shape[1] != 3:
                    img = img.permute(0, 3, 1, 2)  # b h w c -> b c h w
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)
        
        # Concatenate all RGB features
        if len(features) > 0:
            rgb_feat = torch.cat(features, dim=-1)
            # Project to fixed dimension
            rgb_feat = self.rgb_projection(rgb_feat)
            return rgb_feat
        else:
            return None
    
    def forward(self, observations: Dict) -> torch.Tensor:
        """Forward pass without VIB/reconstruction (for inference)."""
        # Check if point cloud is available
        has_point_cloud = self.point_cloud_key in observations and observations[self.point_cloud_key] is not None
        
        # Process point cloud if available
        pn_feat = None
        if has_point_cloud:
            points = observations[self.point_cloud_key]
            assert len(points.shape) == 3, f"Point cloud shape: {points.shape}, length should be 3"
            
            if self.use_imagined_robot:
                img_points = observations[self.imagination_key][..., :points.shape[-1]]
                points = torch.concat([points, img_points], dim=1)
            
            # Get point cloud features
            if self.extractor is not None:
                pn_feat = self.extractor(points)
            else:
                raise RuntimeError("Point cloud extractor is None but point cloud data is provided")
        
        # Process multi-view RGB
        rgb_feat = self.process_rgb_observations(observations)
        
        # Combine features based on strategy
        if has_point_cloud and rgb_feat is not None:
            if self.multiview_integrate_strategy == 'add':
                combined_feat = pn_feat + rgb_feat
            elif self.multiview_integrate_strategy == 'concat':
                combined_feat = torch.cat([pn_feat, rgb_feat], dim=-1)
            else:
                raise NotImplementedError(f"Integration strategy: {self.multiview_integrate_strategy}")
        elif has_point_cloud:
            combined_feat = pn_feat
        elif rgb_feat is not None:
            combined_feat = rgb_feat
        else:
            raise ValueError("No features available: neither point cloud nor RGB observations found")
        
        # Add agent position if used
        if self.use_agent_pos:
            state = observations[self.state_key]
            state_feat = self.state_mlp(state)
            final_feat = torch.cat([combined_feat, state_feat], dim=-1)
        else:
            final_feat = combined_feat
        
        # Apply VIB if enabled
        if self.use_vib:
            if self.multiview_integrate_strategy == 'concat' and rgb_feat is not None:
                # Split features for VIB
                latent_features = []
                
                # Apply VIB to point cloud if available
                if has_point_cloud and pn_feat is not None:
                    mu_pc = self.latent_mu_pc(pn_feat)
                    logvar_pc = self.latent_logvar_pc(pn_feat)
                    if self.training or self.force_stochastic:
                        pc_latent = self._reparameterize(mu_pc, logvar_pc)
                    else:
                        pc_latent = mu_pc
                    latent_features.append(pc_latent)

                # Apply VIB to RGB features
                if rgb_feat is not None:
                    mu_rgb = self.latent_mu_multiview(rgb_feat)
                    logvar_rgb = self.latent_logvar_multiview(rgb_feat)
                    if self.training or self.force_stochastic:
                        rgb_latent = self._reparameterize(mu_rgb, logvar_rgb)
                    else:
                        rgb_latent = mu_rgb
                    latent_features.append(rgb_latent)

                # Apply VIB to state if used
                if self.use_agent_pos:
                    mu_state = self.latent_mu_state(state_feat)
                    logvar_state = self.latent_logvar_state(state_feat)
                    if self.training or self.force_stochastic:
                        state_latent = self._reparameterize(mu_state, logvar_state)
                    else:
                        state_latent = mu_state
                    latent_features.append(state_latent)

                # Concatenate all latent features
                final_feat = torch.cat(latent_features, dim=-1)
            else:
                # Use parent class VIB logic for non-concat strategies
                pc_feat = combined_feat[:, :self.out_channel]
                mu_pc = self.latent_mu_pc(pc_feat)
                logvar_pc = self.latent_logvar_pc(pc_feat)
                if self.training or self.force_stochastic:
                    pc_latent = self._reparameterize(mu_pc, logvar_pc)
                else:
                    pc_latent = mu_pc

                if self.use_agent_pos:
                    state_feat = final_feat[:, self.out_channel:]
                    mu_state = self.latent_mu_state(state_feat)
                    logvar_state = self.latent_logvar_state(state_feat)
                    if self.training or self.force_stochastic:
                        state_latent = self._reparameterize(mu_state, logvar_state)
                    else:
                        state_latent = mu_state
                    final_feat = torch.cat([pc_latent, state_latent], dim=-1)
                else:
                    final_feat = pc_latent
        
        # Ensure output is 2D tensor [batch_size, feature_dim]
        if len(final_feat.shape) > 2:
            final_feat = final_feat.view(final_feat.shape[0], -1)
        return final_feat
    
    def Recon_VIB_loss(self, observations: Dict):
        """Compute VIB and reconstruction losses with multi-view support."""
        # Check if point cloud is available
        has_point_cloud = self.point_cloud_key in observations and observations[self.point_cloud_key] is not None
        
        # Process point cloud if available
        pn_feat = None
        if has_point_cloud:
            points = observations[self.point_cloud_key]
            assert len(points.shape) == 3
            
            if self.use_imagined_robot:
                img_points = observations[self.imagination_key][..., :points.shape[-1]]
                points = torch.concat([points, img_points], dim=1)
            
            # Get point cloud features
            if self.extractor is not None:
                pn_feat = self.extractor(points)
            else:
                raise RuntimeError("Point cloud extractor is None but point cloud data is provided")
        
        # Process RGB features
        rgb_feat = self.process_rgb_observations(observations)
        
        outputs = {}
        loss_items = {}
        
        # Handle different integration strategies
        if self.multiview_integrate_strategy == 'concat' and rgb_feat is not None:
            # Separate handling for concatenated features
            if self.use_agent_pos:
                state = observations[self.state_key]
                state_feat = self.state_mlp(state)
            else:
                state_feat = None
            
            # Apply VIB to each component
            if self.use_vib:
                latent_list = []
                mu_dict = {}
                logvar_dict = {}
                
                # Point cloud VIB (if available)
                if has_point_cloud and pn_feat is not None:
                    mu_pc = self.latent_mu_pc(pn_feat)
                    logvar_pc = self.latent_logvar_pc(pn_feat)
                    if self.training or self.force_stochastic:
                        pc_latent = self._reparameterize(mu_pc, logvar_pc)
                    else:
                        pc_latent = mu_pc
                    latent_list.append(pc_latent)
                    mu_dict["mu_pc"] = mu_pc
                    logvar_dict["logvar_pc"] = logvar_pc
                    outputs["pc_latent"] = pc_latent
                else:
                    mu_pc = None
                    logvar_pc = None
                    pc_latent = None

                # RGB VIB
                if rgb_feat is not None:
                    mu_rgb = self.latent_mu_multiview(rgb_feat)
                    logvar_rgb = self.latent_logvar_multiview(rgb_feat)
                    if self.training or self.force_stochastic:
                        rgb_latent = self._reparameterize(mu_rgb, logvar_rgb)
                    else:
                        rgb_latent = mu_rgb
                    latent_list.append(rgb_latent)
                    mu_dict["mu_rgb"] = mu_rgb
                    logvar_dict["logvar_rgb"] = logvar_rgb
                    outputs["rgb_latent"] = rgb_latent
                else:
                    mu_rgb = None
                    logvar_rgb = None
                    rgb_latent = None

                # State VIB
                if self.use_agent_pos and state_feat is not None:
                    mu_state = self.latent_mu_state(state_feat)
                    logvar_state = self.latent_logvar_state(state_feat)
                    if self.training or self.force_stochastic:
                        state_latent = self._reparameterize(mu_state, logvar_state)
                    else:
                        state_latent = mu_state
                    latent_list.append(state_latent)
                    mu_dict["mu_state"] = mu_state
                    logvar_dict["logvar_state"] = logvar_state
                    outputs["state_latent"] = state_latent
                else:
                    mu_state = None
                    logvar_state = None
                    state_latent = None
                    outputs["state_latent"] = None
                
                # Concatenate all available latents
                updated_feat = torch.cat(latent_list, dim=-1)
                
                # Update outputs with mu and logvar
                outputs.update(mu_dict)
                outputs.update(logvar_dict)
                
                # Compute KL losses
                kl_loss = 0.0
                if mu_pc is not None:
                    kl_pc = -0.5 * torch.sum(
                        1 + logvar_pc - mu_pc.pow(2) - logvar_pc.exp(), dim=-1
                    ).mean()
                    kl_loss = kl_loss + kl_pc
                
                if mu_rgb is not None:
                    kl_rgb = -0.5 * torch.sum(
                        1 + logvar_rgb - mu_rgb.pow(2) - logvar_rgb.exp(), dim=-1
                    ).mean()
                    kl_loss = kl_loss + kl_rgb
                
                if mu_state is not None:
                    kl_state = -0.5 * torch.sum(
                        1 + logvar_state - mu_state.pow(2) - logvar_state.exp(), dim=-1
                    ).mean()
                    kl_loss = kl_loss + kl_state
                
                kl_loss = self.beta_kl * kl_loss
            else:
                outputs.update({
                    "pc_latent": pn_feat,
                    "rgb_latent": rgb_feat,
                    "state_latent": state_feat
                })
                updated_feat = torch.cat([
                    pn_feat, rgb_feat, 
                    state_feat if state_feat is not None else []
                ], dim=-1)
                kl_loss = torch.tensor(0.0, device=pn_feat.device)
        else:
            # Use parent class logic for add strategy
            if rgb_feat is not None:
                if self.multiview_integrate_strategy == 'add':
                    combined_feat = pn_feat + rgb_feat
                else:
                    raise NotImplementedError(f"Integration strategy: {self.multiview_integrate_strategy}")
            else:
                combined_feat = pn_feat
            
            # Continue with parent class logic
            if self.use_agent_pos:
                state = observations[self.state_key]
                state_feat = self.state_mlp(state)
                final_feat = torch.cat([combined_feat, state_feat], dim=-1)
            else:
                state_feat = None
                final_feat = combined_feat
            
            # Apply VIB
            if self.use_vib:
                pc_feat = combined_feat
                mu_pc = self.latent_mu_pc(pc_feat)
                logvar_pc = self.latent_logvar_pc(pc_feat)
                if self.training or self.force_stochastic:
                    pc_latent = self._reparameterize(mu_pc, logvar_pc)
                else:
                    pc_latent = mu_pc

                if self.use_agent_pos:
                    mu_state = self.latent_mu_state(state_feat)
                    logvar_state = self.latent_logvar_state(state_feat)
                    if self.training or self.force_stochastic:
                        state_latent = self._reparameterize(mu_state, logvar_state)
                    else:
                        state_latent = mu_state
                    updated_feat = torch.cat([pc_latent, state_latent], dim=-1)
                else:
                    mu_state = None
                    logvar_state = None
                    state_latent = None
                    updated_feat = pc_latent

                outputs.update({
                    "pc_latent": pc_latent,
                    "state_latent": state_latent,
                    "mu_pc": mu_pc,
                    "logvar_pc": logvar_pc,
                    "mu_state": mu_state,
                    "logvar_state": logvar_state,
                })

                # Compute KL loss
                kl_pc = -0.5 * torch.sum(
                    1 + logvar_pc - mu_pc.pow(2) - logvar_pc.exp(), dim=-1
                ).mean()

                if self.use_agent_pos:
                    kl_state = -0.5 * torch.sum(
                        1 + logvar_state - mu_state.pow(2) - logvar_state.exp(), dim=-1
                    ).mean()
                    kl_loss = self.beta_kl * (kl_pc + kl_state)
                else:
                    kl_loss = self.beta_kl * kl_pc
            else:
                outputs.update({
                    "pc_latent": combined_feat,
                    "state_latent": state_feat
                })
                updated_feat = final_feat
                kl_loss = torch.tensor(0.0, device=final_feat.device)
        
        loss_items["kl_loss"] = kl_loss.mean().item()
        
        # Reconstruction losses
        if self.use_recon:
            recon_loss = 0.0
            
            # Point cloud reconstruction (if available)
            if has_point_cloud and outputs.get("pc_latent") is not None:
                recon_pc = self.pc_decoder(outputs["pc_latent"]).reshape(-1, observations['point_cloud'].size(1), 3)
                outputs["recon_pc"] = recon_pc
                
                # Compute reconstruction loss
                assert not self.use_imagined_robot, "Please re-compute ground truth point cloud if using imagined robot"
                recon_pc_loss, _ = chamfer_distance(outputs["recon_pc"], observations['point_cloud'])
                recon_loss = recon_loss + recon_pc_loss
            
            # State reconstruction (if used)
            if self.use_agent_pos and outputs.get("state_latent") is not None:
                # For RGB-only case, we might need to use a different decoder
                if has_point_cloud:
                    recon_state = self.state_decoder(outputs["state_latent"])
                else:
                    # Use the combined feature decoder for RGB-only case
                    recon_state = self.state_decoder(outputs["state_latent"])
                outputs["recon_state"] = recon_state
                recon_state_loss = F.mse_loss(outputs["recon_state"], observations['agent_pos'])
                recon_loss = recon_loss + 1.0 * recon_state_loss
            
            recon_loss = self.beta_recon * recon_loss
        else:
            recon_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        loss_items["recon_loss"] = recon_loss.mean().item()
        loss = kl_loss + recon_loss
        
        return loss, loss_items, updated_feat
    
    @torch.no_grad()
    def output_shape(self):
        """Compute output shape of the encoder."""
        return self.n_output_channels


class DP3ResNetEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 backbone=None,
                 integrate_strategy='add', # add, concat features of point cloud and image
                 use_agent_pos=True,
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        self.use_agent_pos = use_agent_pos
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
        self.backbone_name = backbone
        self.integrate_strategy = integrate_strategy
        if backbone is not None:   
            assert backbone in ["resnet50", "resnet18", "clip"]
                # Frozen backbone
            if backbone == "resnet50":
                self.backbone, self.image_normalize = load_resnet50()
                self.resnet_fc = nn.Linear(2048, out_channel)
            elif backbone == "resnet18":
                self.backbone, self.image_normalize = load_resnet18()
                self.resnet_fc = nn.Linear(2048, out_channel)
            elif backbone == "clip":
                cprint("Load CLIP model", "yellow")
                self.backbone, self.image_normalize = load_clip()
            for p in self.backbone.parameters():
                p.requires_grad = False
            
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape if self.point_cloud_shape else 'None (RGB-only mode)'}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointResNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointResNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]
        
        if use_agent_pos:
            self.n_output_channels  += output_dim
        if backbone is not None and self.integrate_strategy == 'concat':
            self.n_output_channels += out_channel
        
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))
        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        if len(points.shape) != 3:
            print(f"[WARNING] point cloud shape: {points.shape}, expected 3D, got {len(points.shape)}D")
        assert len(points.shape) == 3, f"point cloud shape: {points.shape}, length should be 3"
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        pn_feat = self.extractor(points)    # B * out_channel
        
        # if use 2d image feature
        if self.backbone_name is not None:
            img = observations[self.rgb_image_key]
            # import pdb; pdb.set_trace()
            if img.shape[1] != 3:
                img = einops.rearrange(img, "b h w c -> b c h w")
            img = self.image_normalize(img)
            img_feat = self.backbone(img)
            img_feat = self.resnet_fc(img_feat['res6'])
            if self.integrate_strategy == 'add':
                pn_feat = pn_feat + img_feat
            elif self.integrate_strategy == 'concat':
                pn_feat = torch.cat([pn_feat, img_feat], dim=-1)
            else:
                raise NotImplementedError(f"integrate_strategy: {self.integrate_strategy}")
        if self.use_agent_pos:
            state = observations[self.state_key]
            state_feat = self.state_mlp(state)  # B * 64
            final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        else:
            final_feat = pn_feat
        return final_feat


    def output_shape(self):
        return self.n_output_channels
class iDP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='dp3_encoder',
                 point_downsample=True,
                 ):
        super().__init__()
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.n_output_channels = pointcloud_encoder_cfg.out_channels
        
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]

        self.num_points = pointcloud_encoder_cfg.num_points # 4096
        


        cprint(f"[iDP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[iDP3Encoder] state shape: {self.state_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        
        self.downsample = point_downsample
        if self.downsample:
            self.point_preprocess = point_process.uniform_sampling_torch
        else:
            self.point_preprocess = nn.Identity()
        
        
        
        if pointnet_type == "multi_stage_pointnet":
            from .multi_stage_pointnet import MultiStagePointNetEncoder
            self.extractor = MultiStagePointNetEncoder(out_channels=pointcloud_encoder_cfg.out_channels)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")

        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        if self.downsample:
            points = self.point_preprocess(points, self.num_points)
           
        pn_feat = self.extractor(points)    # B * out_channel
         
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels
class ViTEncoder(nn.Module):
    """With ViT backbone"""

    def __init__(
        self,
        backbone,
        observation_space,
        img_cond_steps=1,
        num_obs_steps=1,
        spatial_emb=0,
        visual_feature_dim=128,
        dropout=0,
        num_img=1,
        augment=False,
        use_agent_pos=False,
        image_his_cat=False, # concatenate image history features or use timestep * channel as input
        shape_meta=None,
    ):
        super().__init__()
        self.image_his_cat = image_his_cat 
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.use_agent_pos = use_agent_pos
        self.shape_meta = shape_meta
        self.state_shape = observation_space[self.state_key]
        cond_dim = self.state_shape[0] * num_obs_steps
        # vision
        self.backbone = backbone
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        self.num_obs_steps = num_obs_steps
        if spatial_emb > 0:
            assert spatial_emb > 1, "this is the dimension"
            if num_img > 1:
                self.compress1 = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                self.compress2 = deepcopy(self.compress1)
            else:  # TODO: clean up
                self.compress = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
            visual_feature_dim = spatial_emb * num_img
        else:
            self.compress = nn.Sequential(
                nn.Linear(self.backbone.repr_dim, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )

    def forward(
        self,
        cond: dict,
        **kwargs,
    ):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B * To, Do)
            rgb: (B * To, C, H, W)

        TODO long term: more flexible handling of cond
        """
        if cond[self.rgb_image_key].shape[1] != 3:
            cond[self.rgb_image_key] = einops.rearrange(cond[self.rgb_image_key], "b h w c -> b c h w")
        B = int(cond[self.rgb_image_key].shape[0] / self.num_obs_steps)

        Ta, T_rgb = self.num_obs_steps, self.num_obs_steps
        C, H, W = cond[self.rgb_image_key].shape[-3:]
        cond = dict_apply(cond, lambda x: x.reshape(B, self.num_obs_steps, *x.shape[1:])) # [2 i.e. batch_size * n_obs, 512, 3], [2, 24] 
        # flatten history
        state = cond[self.state_key].view(B, -1)

        # Take recent images --- sometimes we want to use fewer img_cond_steps than cond_steps (e.g., 1 image but 3 prio)
        rgb = cond[self.rgb_image_key][:, -self.img_cond_steps :]

        # concatenate images in cond by channels
        if self.num_img > 1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")
        # convert rgb to float32 for augmentation
        rgb = rgb.float()

        # get vit output - pass in two images separately
        if self.num_img > 1:  # TODO: properly handle multiple images
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)
            feat1 = self.compress1.forward(feat1, state)
            feat2 = self.compress2.forward(feat2, state)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:  # single image
            if self.augment:
                rgb = self.aug(rgb)
            feat = self.backbone(rgb)

            # compress
            if isinstance(self.compress, SpatialEmb):
                feat = self.compress.forward(feat, state)
            else:
                feat = feat.flatten(1, -1)
                feat = self.compress(feat)
        cond_encoded = torch.cat([feat, state], dim=-1)

        return cond_encoded
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    @torch.no_grad()
    def output_shape(self,):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            # if key == self.rgb_image_key:
            #     shape = (3, 96, 96)
            this_obs = torch.zeros(
                (self.num_obs_steps, ) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        # assert example_output.shape[0] == 1
        # import pdb; pdb.set_trace()
        return example_output.shape[-1]
