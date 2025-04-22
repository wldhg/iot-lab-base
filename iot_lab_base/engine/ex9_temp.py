import os
from typing import override

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from ..db import AppendOnlyDB
from .abc import InferenceEngine


class Ex9Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, hidden_layers: int):
        super().__init__()

        layers = []
        for i in range(hidden_layers):
            layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(hidden_size, 1))
        self.fc = nn.Sequential(*layers)
        self.fc.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Ex9TempEngine(InferenceEngine):
    @override
    def __init__(self, db: AppendOnlyDB):
        self.db = db
        self.model = Ex9Model(36, 32, 3)
        self.model.eval()

        state_dict = torch.load(
            os.path.join(os.path.dirname(__file__), "..", "..", "ex9_temp.ckpt")
        )
        self.model.load_state_dict(state_dict)

    @override
    def infer(self) -> float:
        data = self.db.get_last_n("temperature", 36)
        if len(data) < 36:
            raise ValueError("Not enough data for inference")

        # Interpolate the last 36 entries
        times = np.array([entry.ts for entry in data], dtype=np.float32)
        values = np.array([entry.value for entry in data], dtype=np.float32)
        new_times = np.linspace(times[-1] - 5 * 36, times[-1], 36)
        new_values = np.interp(new_times, times, values)

        # Make it a tensor
        values_tensor = torch.from_numpy(new_values).unsqueeze(0).float()

        # Inference
        output: torch.Tensor = self.model(values_tensor)

        return output.squeeze().item()  # Convert to float

    @property
    @override
    def model_name(self) -> str:
        return "ex9_temp"
