import torch
from torch import nn
import torch.nn.functional as F

"""
L(w1`, w2`) = [max(margin * dh(w1, w2) - d(w1`, w2`), 0)] + [max(d(w1`, w2`) - margin * (dh(w1, w2) + 1), 0)]

where 
    w` - output of neural network with given "w" word
    d(w1`, w2`) = squared l2 norm of (w1 - w2) 
    dh(w1, w2) - string distance between words w1, w2
"""

class SyntaxLoss(nn.Module):
    def __init__(self, margin: float = 0.2, device = "cuda"):
        super().__init__()
        self._margin = torch.tensor(margin, dtype=torch.float32).to(device)
    
    def forward(
            self,
            first_prediction: torch.Tensor,
            secord_prediction: torch.Tensor,
            string_distance: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates Hamming loss function between to word image encodings.

        Args:
            first_prediction (torch.Tensor): The prediction of the first word. Tensor of shape (N, encoding_dim), where N - batch size
            second_prediction (torch.Tensor): The prediction of the second word. Tensor of shape (N, encoding_dim), where N - batch size
            string_distance (torch.Tensor): The string distance (similarity) between two words. Tensor of shape (N,)
            margin (float): margin distance between encodings

        Returns:
            (torch.Tensor): Tensor of shape (1,)
        """
        # squared_norm = torch.linalg.vector_norm(first_prediction - secord_prediction, dim=1, ord=2) ** 2
        norm = F.pairwise_distance(first_prediction, secord_prediction)
        left_part = F.relu(self._margin * string_distance - norm)
        right_part = F.relu(norm - self._margin * (string_distance + 1))

        return torch.mean(left_part + right_part)
