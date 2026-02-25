import torch
import torch.nn as nn

class BaseEncoder:

    def __init__(self, input_dim, include_input, funcs, num_freq, max_freq):
        freq_bands = 2.**torch.linspace(0., max_freq, num_freq)

        out_dim = input_dim if include_input else 0
        encoding_fn = [(lambda x : x)] if include_input else []

        for freq in freq_bands :
            for func in funcs :
                encoding_fn.append(lambda x, func=func, freq=freq : func(x * freq))
                out_dim += input_dim

        self.out_dim = out_dim
        self.encodings = encoding_fn


    def encode(self, inputs):
        return torch.cat([fn(inputs) for fn in self.encodings], -1)


def get_base_encoder(multires, use_encoding):

    if not use_encoding:
        return nn.Identity(), 3, []

    encoder = BaseEncoder(3, True, [torch.sin, torch.cos], multires, multires-1)

    return (lambda x, encoder=encoder: encoder.encode(x)), encoder.out_dim, []