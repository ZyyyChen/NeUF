import torch
import torch.nn as nn

from utils import get_voxel_vertices


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class HashEncoder(nn.Module):
    def __init__(self, bounding_box, n_levels, n_features_per_level,
                log2_hashmap_size, base_resolution, finest_resolution):
        super(HashEncoder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level
        
        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))
        self.level_weights = nn.Parameter(torch.zeros(n_levels))

        self.embeddings = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size,
                                        self.n_features_per_level) for i in range(n_levels)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

        self.to(device)

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:, 0] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 4] * weights[:, 0][:, None]
        c01 = voxel_embedds[:, 1] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 5] * weights[:, 0][:, None]
        c10 = voxel_embedds[:, 2] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 6] * weights[:, 0][:, None]
        c11 = voxel_embedds[:, 3] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 7] * weights[:, 0][:, None]

        # step 2
        c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
        c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]

        # step 3
        c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]

        return c

    def forward(self, inputs, active_levels=None):
        if active_levels is None:
            active_levels = self.n_levels

        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b ** i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = get_voxel_vertices(
                inputs, self.bounding_box,
                resolution, self.log2_hashmap_size)

            voxel_embedds = self.embeddings[i](hashed_voxel_indices)

            x_embedded = self.trilinear_interp(inputs, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            if i < active_levels:
                gate = torch.sigmoid(self.level_weights[i])
                x_embedded = x_embedded * gate
            else:
                x_embedded = torch.zeros_like(x_embedded)
            x_embedded_all.append(x_embedded)


        # print(inputs, x_embedded_all)
        return torch.cat(x_embedded_all, dim=-1)



def get_hash_encoder(use_encoding, bounding_box, n_levels=12, n_features_per_level=4,
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
    if not use_encoding:
        return nn.Identity(), 3, []

    encoder = HashEncoder(bounding_box, n_levels, n_features_per_level,
                log2_hashmap_size, base_resolution, finest_resolution)

    return encoder, encoder.out_dim, encoder.parameters()
