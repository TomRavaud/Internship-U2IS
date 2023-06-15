import torch


def shannon_entropy(p: torch.Tensor) -> torch.Tensor:
    """
    Apply the Shannon entropy function to a probability distribution
    
    Args:
        p (Tensor): Probability distribution
    
    Returns:
        Tensor: The entropy of the probability distribution
    """
    return -torch.sum(p*torch.log(p), dim=1)

def least_confidence(p: torch.Tensor) -> torch.Tensor:
    """
    Apply the least confidence function to a probability distribution
    
    Args:
        p (Tensor): Probability distribution
    
    Returns:
        Tensor: The least confidence of the probability distribution
    """
    return 1 - torch.max(p, dim=1)[0]

def confidence_margin(p: torch.Tensor) -> torch.Tensor:
    """
    Apply the margin sampling function to a probability distribution
    
    Args:
        p (Tensor): Probability distribution
    
    Returns:
        Tensor: The margin sampling of the probability distribution
    """
    sorted_p, _ = torch.sort(p, dim=1, descending=True)
    return 1 - (sorted_p[:, 0] - sorted_p[:, 1])
