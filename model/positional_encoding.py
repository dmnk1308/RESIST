import torch

def positional_encoding(tensor, num_encoding_functions=6, freq=2.):
    """Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    encoding = []
    frequency_bands = torch.pi * (freq ** torch.linspace(
        -1.0,
        num_encoding_functions - 2,
        num_encoding_functions,
        dtype=tensor.dtype,
        device=tensor.device,
    ))
    
    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)