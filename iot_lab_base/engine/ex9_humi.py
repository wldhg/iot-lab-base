import os
from typing import override

import numpy as np
import torch

from ..db import AppendOnlyDB
from .abc import InferenceEngine
from .ex9_temp import Ex9Model


class Ex9HumiEngine(InferenceEngine):
    @override
    def __init__(self, db: AppendOnlyDB):
        self.db = db
        self.model = Ex9Model(36, 32, 3)
        self.model.eval()

        state_dict = torch.load(
            os.path.join(os.path.dirname(__file__), "..", "..", "ex9_humi.ckpt")
        )
        self.model.load_state_dict(state_dict)

    @override
    def infer(self) -> float:
        data = self.db.get_last_n("humidity", 36)
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
        return "ex9_humi"
