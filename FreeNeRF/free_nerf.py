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
Implementation of vanilla nerf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type

from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from .encoding import FreeNeRFEncoding


@dataclass
class FreeNeRFModelConfig(VanillaModelConfig):
    """Vanilla Model Config"""

    _target: Type = field(default_factory=lambda: FreeNeRFModel)

    position_encoding_num_frequencies: int = 10
    """Number of frequencies for positional encoding"""

    direction_encoding_num_frequencies: int = 4
    """Number of frequencies for directional encoding"""

    T: int = 30000
    """Number of training steps (must equal to max-num-iterations)"""


class FreeNeRFModel(NeRFModel):
    """Free NeRF model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    config: FreeNeRFModelConfig

    def __init__(
        self,
        config: FreeNeRFModelConfig,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            **kwargs,
        )

        self.step: int = 0
    
    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # remove old fields
        del self.field_coarse
        del self.field_fine

        # create new fields
        position_encoding = FreeNeRFEncoding(
            in_dim=3, num_frequencies=self.config.position_encoding_num_frequencies, min_freq_exp=0.0, include_input=True, implementation="tcnn"
        )
        direction_encoding = FreeNeRFEncoding(
            in_dim=3, num_frequencies=self.config.direction_encoding_num_frequencies, min_freq_exp=0.0, include_input=True, implementation="tcnn"
        )

        self.field_coarse = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        self.field_fine = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def set_step(step: int) -> None:
            self.step = step
            self.field_coarse.position_encoding.set_ratio_x(step / self.config.T)
            self.field_coarse.direction_encoding.set_ratio_x(step / self.config.T)

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=set_step,
            ),
        ]
    