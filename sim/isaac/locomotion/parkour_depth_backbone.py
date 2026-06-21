"""Vendored depth-encoder modules for the Extreme-Parkour-Onboard Go2 policy.

Copied verbatim (behaviour-preserving) from the upstream repo
`change-every/Extreme-Parkour-Onboard` -> `rsl_rl/rsl_rl/modules/depth_backbone.py`
so the sim does not need the full rsl_rl/legged_gym install. Only the two classes
the runtime needs are kept; the upstream `import torchvision` (unused by these two)
and `StackDepthEncoder` are dropped.

`vision_weight.pt` is a state_dict (key ``depth_encoder_state_dict``) for
``RecurrentDepthBackbone(DepthOnlyFCBackbone58x87(None, 32, 512), None)`` -- see
[[project_parkour_policy_contract]]. Architecture/shapes verified against the
shipped weights (Conv 1->32 5x5, Conv 32->64 3x3, GRU hidden 512, output 34).
"""

import torch
import torch.nn as nn


class DepthOnlyFCBackbone58x87(nn.Module):
    """Conv stack mapping a [B, 58, 87] depth image to a 32-dim latent."""

    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim),
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)
        return latent


class RecurrentDepthBackbone(nn.Module):
    """Depth conv backbone + GRU producing a [B, 34] = depth_latent(32) + yaw(2).

    Maintains ``hidden_states`` across calls (one GRU step per call), so the
    caller MUST persist this module instance and reset it (``hidden_states=None``)
    when the episode/robot resets.
    """

    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        if env_cfg is None:
            self.combination_mlp = nn.Sequential(
                nn.Linear(32 + 53, 128),
                activation,
                nn.Linear(128, 32),
            )
        else:
            self.combination_mlp = nn.Sequential(
                nn.Linear(32 + env_cfg.env.n_proprio, 128),
                activation,
                nn.Linear(128, 32),
            )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
            nn.Linear(512, 32 + 2),
            last_activation,
        )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        return depth_latent

    def detach_hidden_states(self):
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.detach().clone()
