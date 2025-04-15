from abc import abstractmethod

from ..db import AppendOnlyDB


class InferenceEngine:
    @abstractmethod
    def __init__(self, db: AppendOnlyDB):
        """
        Initialize the inference engine with the given database.

        db: AppendOnlyDB instance for data storage.
        """
        raise NotImplementedError("Inference engine is not implemented yet.")

    @abstractmethod
    def infer(self) -> float:
        """
        Perform inference on the given data using the specified model.
        This function should load the model from the checkpoint and perform inference
        using the data stored in the database.
        The function should return the inference result as a float.
        """
        raise NotImplementedError("Inference engine is not implemented yet.")

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return the name of the model used for inference.
        """
        raise NotImplementedError("Inference engine is not implemented yet.")
