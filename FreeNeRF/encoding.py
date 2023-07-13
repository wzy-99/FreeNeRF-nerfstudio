# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Encoding functions
"""

from typing import Literal, Optional, Sequence
import torch
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn
from nerfstudio.field_components.encodings import NeRFEncoding


class FreeNeRFEncoding(NeRFEncoding):
    """Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        include_input: bool = False,
        max_freq_exp: float = None,
        implementation: Literal["tcnn", "torch"] = "torch",
    ) -> None:
        assert include_input is True, "FreeNeRF must include input"

        if max_freq_exp is None:
            max_freq_exp = num_frequencies - 1
        else:
            assert max_freq_exp == num_frequencies - 1, "FreeNeRF must max_freq_exp == num_frequencies - 1"

        super().__init__(in_dim, num_frequencies, min_freq_exp, max_freq_exp, include_input, implementation)
        self.ratio_x: float = 1.0

    def forward(
        self, 
        in_tensor: Float[Tensor, "*bs input_dim"], 
        covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None,
    ) -> Float[Tensor, "*bs output_dim"]:
        if self.tcnn_encoding is not None:
            shape = in_tensor.shape
            if len(shape) > 2:
                in_tensor = in_tensor.reshape(-1, shape[-1])
            encodings = self.tcnn_encoding(in_tensor) # [sinx, cosx, sin2x, cos2x, ..., siny, cosy, sin2y, cos2y, ...]
            weights = self.get_weights().to(encodings.device)
            encodings = encodings * weights
            encodings = torch.cat((encodings, in_tensor), dim=-1) # [sinx, cosx, sin2x, cos2x, ..., siny, cosy, sin2y, cos2y, ..., x, y]
            if len(shape) > 2:
                encodings = encodings.reshape(*shape[:-1], -1)
        else:
            encodings = self.pytorch_fwd(in_tensor, covs) # [sinx, sin2x, ..., siny, sin2y, ..., cosx, cos2x, ..., cosy, cos2y, ..., x, y]
            weights = self.get_weights().to(encodings.device)
            encodings[:, :-self.in_dim] = encodings[:,:-self.in_dim]* weights

        return encodings

    @property
    def L(self) -> int:
        return self.num_frequencies
    
    @torch.no_grad()
    def get_weights(self) -> Float[Tensor, "1 output_dim"]:
        weights = torch.ones((self.L,), dtype=torch.float32) # [sinx, sin2x, sin3x, ..., sinLx]
        weights[int(self.ratio_x * self.L) : int(self.ratio_x * self.L) + 1] = self.ratio_x * self.L - int(self.ratio_x * self.L)
        weights[int(self.ratio_x * self.L) + 1 : ] = 0.0
        if self.tcnn_encoding is not None:
            weights = weights.unsqueeze(-1)
            weights = weights.repeat((self.in_dim, 2)).reshape(-1) # [sinx, cosx, sin2x, cos2x, ..., siny, cosy, sin2y, cos2y, ...]
        else:
            weights = weights.unsqueeze(0)
            weights = weights.repeat((2, self.in_dim)).reshape(-1) # [sinx, sin2x, ..., siny, sin2y, ..., cosx, cos2x, ..., cosy, cos2y, ...]
        return weights
    
    def set_ratio_x(self, ratio_x: float) -> None:
        ratio_x = max(0.0, min(1.0, ratio_x))
        self.ratio_x = ratio_x