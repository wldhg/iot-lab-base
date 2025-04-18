from typing import override

import numpy as np
import torch
import torch.nn as nn

from ..db import AppendOnlyDB
from .abc import InferenceEngine


class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # Example model with a single linear layer

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ExampleInferenceEngine(InferenceEngine):
    @override
    def __init__(self, db: AppendOnlyDB):
        self.db = db
        self.model = ExampleModel()
        # NOTE: You may want to load a checkpoint here

    @override
    def infer(self) -> float:
        # Example inference logic
        # Load data from the database and perform inference
        data = self.db.get_last_n("temperature", 10)
        if len(data) < 10:
            raise ValueError("Not enough data for inference")

        data_np = np.array([entry.value for entry in data], dtype=np.float32)
        data_tensor = torch.from_numpy(data_np).unsqueeze(0)

        output: torch.Tensor = self.model(data_tensor)

        return output.squeeze().item()  # Convert to float

    @property
    @override
    def model_name(self) -> str:
        return "example_model"
