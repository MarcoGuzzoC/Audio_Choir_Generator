from dataclasses import dataclass
import torch


@dataclass
class GREGOConfig:
    """
    Data class that stores the configuration for a GREGO model.
    TODO
    """

    # data
    batch_size: int = 18

    # model
    n_embed: int = 216
    block_size: int = 216
    dropout: float = 0.2
    bias: bool = False
    number_of_stack: int = 6

    # Training Loop
    max_epochs: int = 1000

    # adamw optimizer
    learning_rate: float = 0.00001

    # system
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __str__(self):
        categories = {
            "Data": ["batch_size"],
            "Model": ["n_embed", "block_size", "dropout", "bias"],
            "Training Loop": ["max_epochs"],
            "AdamW Optimizer": ["learning_rate"],
            "System": ["device"],
        }

        output = "TransformerConfig:\n"
        for category, attributes in categories.items():
            output += f"\n{category}:\n"

            max_attr_length = max(len(attr) for attr in attributes)
            max_value_length = max(len(str(getattr(self, attr))) for attr in attributes)

            output += "+" + "-" * (max_attr_length + max_value_length + 5) + "+\n"
            for attr in attributes:
                output += f"| {attr:<{max_attr_length}} : {getattr(self, attr):<{max_value_length}} |\n"
            output += "+" + "-" * (max_attr_length + max_value_length + 5) + "+\n"
        return output
