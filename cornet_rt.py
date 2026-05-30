
from collections import OrderedDict
import torch
from torch import nn
import numpy as np
HASH = '933c001c'
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = False

class Flatten(nn.Module):
    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules.
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    """
    Helper module that stores the current tensor. Useful for accessing by name.
    """

    def forward(self, x):
        return x


class CORblock_RT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, out_shape=None, α=0.9, β=0.5, γ=0.9, noise_scale=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape
        self.noise_scale = noise_scale

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=kernel_size // 2)
        self.norm_input = nn.GroupNorm(32, out_channels)
        self.nonlin_input = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.nonlin1 = nn.ReLU(inplace=True)
        self._last_pre_relu_state = None
        self._last_adaptive_state = None
        self.output = Identity()  # for easy access to this block's output

        self.β = β
        self.γ = γ
        self.α = α

    def forward(self, inp=None, state=None, s=None, timestep=0, output=None, batch_size=None, alpha=0.00001):
        # Ensure all tensors are on the same device as the model
        device = next(self.conv_input.parameters()).device

        if inp is None:  # at t=0, there is no input yet except to V1
            inp = torch.zeros([batch_size, self.out_channels, self.out_shape, self.out_shape], device=device)
        else:
            inp = inp.to(device)
            inp = self.conv_input(inp)
            inp = self.norm_input(inp)
            inp = self.nonlin_input(inp)

        if state is None:  # at t=0, state is initialized to zeros
            state = torch.zeros([batch_size, self.out_channels, self.out_shape, self.out_shape], device=device)

        if timestep == 0:
            s = torch.zeros([batch_size, self.out_channels, self.out_shape, self.out_shape], device=device)

        # Generate Gaussian noise (mean=0, std=1) and scale it
        gaussian_noise = torch.randn(inp.size(), device=device)  # Gaussian noise ~ N(0, 1)
        scaled_random_tensor = gaussian_noise * self.noise_scale * alpha
          # Update adaptive state `s`
        if timestep > 0:
            s = self.α * s + (1 - self.α) * output

        
        if output is None:
            output = torch.zeros_like(inp)
        x_adjusted = inp -(self.β * s)
        #x_adjusted = inp + output  
        x = self.conv1(x_adjusted)
        x = self.norm1(x)

        # Update state
        if timestep == 0:
            state = x + scaled_random_tensor
        else:
            state = x + (1 - self.γ) * state + scaled_random_tensor

        state_before_relu = state.clone()
        self._last_pre_relu_state = state_before_relu
        self._last_adaptive_state = s
        output = self.output(self.nonlin1(state))

        return output, state, s, state_before_relu


class CORnet_RT(nn.Module):
    def __init__(self, times=601):
        super().__init__()
        self.times = times
        self.V1 = CORblock_RT(3, 64, kernel_size=7, stride=4, out_shape=56, α=0.96, β=0., γ=0.6, noise_scale=0.7)
        self.V2 = CORblock_RT(64, 128, stride=2, out_shape=28, α=0.96, β=0., γ=0.5, noise_scale=0.7)
        self.V4 = CORblock_RT(128, 256, stride=2, out_shape=14, α=0.96, β=0., γ=0.4, noise_scale=0.7)
        self.IT = CORblock_RT(256, 512, stride=2, out_shape=7, α=0.96, β=0., γ=0.3, noise_scale=0.7)
    



        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000))
        ]))

    def forward(self, inp, alpha=0.00001):
        # Get the device of the model
        device = next(self.V1.conv_input.parameters()).device

        outputs = {'inp': inp.to(device)}
        states = {}
        adaptive_states = {}
        state_sums = {}
        state_counts = {}
        blocks = ['inp', 'V1', 'V2', 'V4', 'IT']

        # Initialize the outputs, states, and adaptive states for t=0
        for block in blocks[1:]:
            if block == 'V1':  # at t=0 input to V1 is the image
                new_output, new_state, new_s, state_before_relu = getattr(self, block)(
                    inp[:, 0].to(device), batch_size=len(outputs['inp']))
            else:  # at t=0 there is no input yet to V2 and up
                new_output, new_state, new_s, state_before_relu = getattr(self, block)(
                    inp=None, batch_size=len(outputs['inp']))

            # Initialize output, state, and adaptive state tensors
            outputs[block] = torch.zeros((new_output.shape[0], self.times, *new_output.shape[1:]), device=device)
            outputs[block][:, 0, :, :, :] = new_output

            states[block] = torch.zeros((new_state.shape[0], self.times, *new_state.shape[1:]), device=device)
            states[block][:, 0] = state_before_relu

            adaptive_states[block] = torch.zeros((new_output.shape[0], self.times, *new_output.shape[1:]), device=device)
            adaptive_states[block][:, 0] = new_s
    

        # Run the forward pass for all time steps
        for t in range(1, self.times):
            for block in blocks[1:]:
                prev_block = blocks[blocks.index(block) - 1]
                prev_output = outputs[prev_block][:, t if block == 'V1' else t - 1]
                current_prev_output = outputs[block][:, t - 1]
                prev_state = states[block][:, t - 1]
                prev_s = adaptive_states[block][:, t - 1]

                # Run the forward pass of the block
                new_output, new_state, new_s, state_before_relu = getattr(self, block)(
                    inp=prev_output, state=prev_state, s=prev_s, timestep=t, output=current_prev_output,
                    batch_size=len(outputs['inp']), alpha=alpha
                )

                # Update the dictionaries
                outputs[block][:, t, :, :, :] = new_output
                states[block][:, t] = state_before_relu
                adaptive_states[block][:, t] = new_s

        out = self.decoder(outputs['IT'][:, -1])
        return out
