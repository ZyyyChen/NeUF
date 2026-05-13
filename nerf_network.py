import torch
import torch.nn as nn
import torch.nn.functional as F
import base_encoder
import hash_encoder
import dual_freq_encoder
import kronecker_encoder
from datetime import date

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NeRF(nn.Module) :
    def __init__(self, ckpt=None):
        super(NeRF, self).__init__()
        self.encode = None
        self.encode_dirs = None
        self.dir_ch = 0
        self.in_ch = 0
        self.out_ch = 0
        self.skips = []
        self.model = None
        self.sigma_linear = None

        self.use_encoding = True
        self.use_direction = True

        self.encoder_params = []
        self.dir_encoder_params = []

        self.encoding_type = ""

        self.num_freq = 0
        self.num_freq_dir=0

        self.encoding_initialized = False
        self.dual_encoder: dual_freq_encoder.DualFreqEncoder | None = None
        self.kronecker_encoder: kronecker_encoder.KroneckerHashPE | None = None
        self.training_progress: float = 0.0

        if ckpt != None :
            self._init_from_ckpt(ckpt)


    def _init_from_ckpt(self, ckpt):

        if(ckpt["encoding"] == "FREQ") :
            self.init_base_encoding(
                use_directions=ckpt["use_directions"],
                use_encoding=ckpt["use_encoding"],
                num_freq=ckpt.get("num_freq",0),
                num_freq_dir=ckpt.get("num_freq_dir",0)
            )

        elif(ckpt["encoding"] == "HASH") :
            self.init_hash_encoding(
                use_directions=ckpt["use_directions"],
                use_encoding=ckpt["use_encoding"],
                bounding_box=ckpt.get("bounding_box",None),
                n_levels=ckpt.get("n_levels",0),
                n_features_per_level=ckpt.get("n_features_per_level",0),
                log2_hashmap_size=ckpt.get("log2_hashmap_size",0),
                base_resolution=ckpt.get("base_resolution",0),
                finest_resolution=ckpt.get("finest_resolution",0)
            )
            if self.use_encoding :
                self.encode.load_state_dict(ckpt["hash_encoder_state"])
        elif ckpt["encoding"].startswith("DUAL_"):
            encoding_name = ckpt["encoding"].upper()
            pe_type = "fourier" if encoding_name in {"DUAL_FREQ", "DUAL_FOURIER"} else "hash"
            self.init_dual_encoding(
                pe_type=pe_type,
                bounding_box=ckpt.get("bounding_box"),
                n_levels_low=ckpt.get("n_levels_low", 8) or 8,
                n_levels_high=ckpt.get("n_levels_high", 8) or 8,
                n_features_per_level=ckpt.get("n_features_per_level", 2) or 2,
                log2_hashmap_size=ckpt.get("log2_hashmap_size", 19) or 19,
                base_resolution_low=ckpt.get("base_resolution_low", 16) or 16,
                finest_resolution_low=ckpt.get("finest_resolution_low", 64) or 64,
                base_resolution_high=ckpt.get("base_resolution_high", 64) or 64,
                finest_resolution_high=ckpt.get("finest_resolution_high", 512) or 512,
                sigma_low=ckpt.get("sigma_low", 1.0) or 1.0,
                sigma_high=ckpt.get("sigma_high", 20.0) or 20.0,
                n_freq=ckpt.get("n_freq", 64) or 64,
                use_gate=ckpt.get("use_gate", True),
                hf_activate_ratio=ckpt.get("hf_activate_ratio", 0.6) or 0.6,
                hf_max_weight=ckpt.get("hf_max_weight", 1.0) or 1.0,
            )
            if self.dual_encoder is not None and "dual_encoder_state" in ckpt:
                self.dual_encoder.load_state_dict(ckpt["dual_encoder_state"])
        elif ckpt["encoding"] == "KRONECKER":
            self.init_kronecker_encoding(
                bounding_box=ckpt.get("bounding_box"),
                n_levels_lateral=ckpt.get("n_levels_lateral", 8) or 8,
                n_levels_axial=ckpt.get("n_levels_axial", 8) or 8,
                finest_lateral=ckpt.get("finest_lateral", 128) or 128,
                finest_axial=ckpt.get("finest_axial", 512) or 512,
                n_features_per_level=ckpt.get("n_features_per_level", 2) or 2,
                log2_hashmap_size=ckpt.get("log2_hashmap_size", 19) or 19,
                base_resolution=ckpt.get("base_resolution", 16) or 16,
                combine=ckpt.get("combine", "cat") or "cat",
            )
            if self.kronecker_encoder is not None and "kronecker_encoder_state" in ckpt:
                self.kronecker_encoder.load_state_dict(ckpt["kronecker_encoder_state"])
        else:
            print("unknown model type:",ckpt["encoding"])
            exit(-1)

        self.init_model()
        incompatible = self.load_state_dict(ckpt["network_fn_state_dict"], strict=False)
        expected_missing = {"sigma_linear.weight", "sigma_linear.bias"}
        unexpected_missing = set(incompatible.missing_keys) - expected_missing
        if unexpected_missing or incompatible.unexpected_keys:
            print(
                "Checkpoint loaded with incompatible keys: "
                f"missing={sorted(unexpected_missing)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )


    def init_hash_encoding(self, bounding_box, n_levels=16, n_features_per_level=2,
                       log2_hashmap_size=19, base_resolution=16, finest_resolution=256, use_directions=True, use_encoding=True):
        if self.encoding_initialized :
            print("encoding initialized twice")
            exit(-1)

        self.encode, self.in_ch, self.encoder_params = hash_encoder.get_hash_encoder(use_encoding, bounding_box, n_levels, n_features_per_level, log2_hashmap_size, base_resolution, finest_resolution)
        if use_directions:
            self.encode_dirs, self.dir_ch, self.dir_encoder_params = base_encoder.get_base_encoder(4,
                                                                                              use_encoding)

        self.out_ch = 1
        self.skips = [4]

        self.use_encoding = use_encoding
        self.use_direction = use_directions

        self.encoding_type = "HASH"
        self.encoding_initialized = True



    def init_base_encoding(self,use_directions=True, use_encoding=True, num_freq=10, num_freq_dir=4):
        if self.encoding_initialized :
            print("encoding initialized twice")
            exit(-1)

        self.encode, self.in_ch, self.encoder_params = base_encoder.get_base_encoder(num_freq, use_encoding)
        if use_directions :
            self.encode_dirs, self.dir_ch, self.dir_encoder_params = base_encoder.get_base_encoder(num_freq_dir, use_encoding)


        self.out_ch = 1
        self.skips = [4]

        self.num_freq = num_freq
        self.num_freq_dir = num_freq_dir
        self.use_encoding = use_encoding
        self.use_direction = use_directions
        self.encoding_type = "FREQ"
        self.encoding_initialized = True

    def init_dual_encoding(
        self,
        pe_type: str = "hash",
        bounding_box=None,
        n_levels_low: int = 8,
        n_levels_high: int = 8,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution_low: int = 16,
        finest_resolution_low: int = 64,
        base_resolution_high: int = 64,
        finest_resolution_high: int = 512,
        sigma_low: float = 1.0,
        sigma_high: float = 20.0,
        n_freq: int = 64,
        use_gate: bool = True,
        hf_activate_ratio: float = 0.6,
        hf_max_weight: float = 1.0,
    ):
        if self.encoding_initialized :
            print("encoding initialized twice")
            exit(-1)

        pe_type = pe_type.lower()
        self.dual_encoder = dual_freq_encoder.DualFreqEncoder(
            pe_type=pe_type,
            bounding_box=bounding_box,
            n_levels_low=n_levels_low,
            n_levels_high=n_levels_high,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution_low=base_resolution_low,
            finest_resolution_low=finest_resolution_low,
            base_resolution_high=base_resolution_high,
            finest_resolution_high=finest_resolution_high,
            sigma_low=sigma_low,
            sigma_high=sigma_high,
            n_freq=n_freq,
            use_gate=use_gate,
            hf_activate_ratio=hf_activate_ratio,
            hf_max_weight=hf_max_weight,
        )

        self.encode = lambda x: self.dual_encoder(x, self.training_progress)
        self.in_ch = self.dual_encoder.out_dim
        self.out_ch = 1
        self.skips = [4]
        self.dir_ch = 0
        self.use_encoding = True
        self.use_direction = False
        self.encoding_type = "DUAL_FREQ" if pe_type == "fourier" else "DUAL_HASH"
        self.encoder_params = list(self.dual_encoder.parameters())
        self.encoding_initialized = True

    def init_kronecker_encoding(
        self,
        bounding_box,
        n_levels_lateral: int = 8,
        n_levels_axial: int = 8,
        finest_lateral: int = 128,
        finest_axial: int = 512,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        combine: str = "cat",
    ):
        if self.encoding_initialized :
            print("encoding initialized twice")
            exit(-1)

        self.kronecker_encoder = kronecker_encoder.KroneckerHashPE(
            bounding_box=bounding_box,
            n_levels_lateral=n_levels_lateral,
            n_levels_axial=n_levels_axial,
            finest_lateral=finest_lateral,
            finest_axial=finest_axial,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            combine=combine,
        )
        self.encode = self.kronecker_encoder
        self.in_ch = self.kronecker_encoder.out_dim
        self.out_ch = 1
        self.skips = [4]
        self.dir_ch = 0
        self.use_encoding = True
        self.use_direction = False
        self.encoding_type = "KRONECKER"
        self.encoder_params = list(self.kronecker_encoder.parameters())
        self.encoding_initialized = True


    def init_model(self, D=8, W=256):
        if(not self.encoding_initialized):
            print("NeRF: encoding not initialized, use init_*_encoding before init_model")
            exit(-1)

        input_ch = int(self.in_ch)
        input_ch_views = int(self.dir_ch)

        # print("---------------------\nINIT NERF MODEL:\ninputs:",input_ch,"\nview inputs:",input_ch_views,"\nout:", self.out_ch)

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        # Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        # Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if self.use_direction:
            self.feature_linear = nn.Linear(W, W)

        self.output_linear = nn.Linear(W, 1)
        self.sigma_linear = nn.Linear(W, 1)

        self.to(device)


    def forward(self,x):
        input_pts, input_views = torch.split(x, [self.in_ch, self.dir_ch], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_direction:
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)


        i_clean = self.output_linear(h)
        log_sigma = self.sigma_linear(h)

        return i_clean, log_sigma


    def batchify(self, chunk, return_sigma=False):
        if chunk is None:
            return self.model

        def ret(inputs):
            outputs = [
                self.forward(inputs[i:i + chunk])
                for i in range(0, inputs.shape[0], chunk)
            ]
            i_clean = torch.cat([output[0] for output in outputs], 0)
            if not return_sigma:
                return i_clean

            log_sigma = torch.cat([output[1] for output in outputs], 0)
            return i_clean, log_sigma

        return ret

    def query(self, inputs, dirs, netchunk=1024*64, return_sigma=False):
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        if hasattr(self.encode, 'forward') and self.encoding_type == "HASH" and self.use_encoding:
            embedded = self.encode(inputs_flat, active_levels=getattr(self, '_active_levels', None))
        else:
            embedded = self.encode(inputs_flat)

        if self.use_direction :
            input_dirs_flat = torch.reshape(dirs, [-1, dirs.shape[-1]])
            embedded_dirs = self.encode_dirs(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        outputs_flat = self.batchify(netchunk, return_sigma=return_sigma)(embedded)
        # outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs_flat

    def query_with_uncertainty(self, inputs, dirs, netchunk=1024*64):
        return self.query(inputs, dirs, netchunk=netchunk, return_sigma=True)

    def grad_vars(self):
        params = list(self.parameters())
        if self.encoder_params:
            params += list(self.encoder_params)
        if self.dir_encoder_params:
            params += list(self.dir_encoder_params)

        unique_params = []
        seen_param_ids = set()
        for param in params:
            param_id = id(param)
            if param_id in seen_param_ids:
                continue
            seen_param_ids.add(param_id)
            unique_params.append(param)

        return unique_params

    def get_encode_name(self):
        return self.encoding_type if self.use_encoding else "NONE"

    def get_rep_name(self):
        # d = date.today().strftime("%d-%m-%Y")
        e = self.get_encode_name()
        dir = "_dirs" if self.use_direction else ""

        return e+dir

    def get_save_dict(self):
        dic = {
                "encoding": self.encoding_type,
                "use_directions": self.use_direction,
                "use_encoding": self.use_encoding,
                "network_fn_state_dict": self.state_dict()
            }
        if self.use_encoding :
            if self.encoding_type == "FREQ" :
                dic.update({
                    #"bounding_box": self.encode.bounding_box,
                    "num_freq": self.num_freq,
                    "num_freq_dir": self.num_freq_dir
                })
            elif self.encoding_type == "HASH" :
                dic.update({
                    "bounding_box": self.encode.bounding_box,
                    "n_levels": self.encode.n_levels,
                    "n_features_per_level": self.encode.n_features_per_level,
                    "log2_hashmap_size": self.encode.log2_hashmap_size,
                    "base_resolution": float(self.encode.base_resolution.cpu()),
                    "finest_resolution": float(self.encode.finest_resolution.cpu()),
                    "hash_encoder_state": self.encode.state_dict(),
                })
            elif self.encoding_type == "KRONECKER" and self.kronecker_encoder is not None:
                enc = self.kronecker_encoder
                dic.update({
                    "bounding_box": enc.bounding_box,
                    "n_levels_lateral": enc.enc_xy.n_levels,
                    "n_levels_axial": enc.enc_xz.n_levels,
                    "finest_lateral": float(enc.enc_xy.finest_resolution.cpu()),
                    "finest_axial": float(enc.enc_xz.finest_resolution.cpu()),
                    "n_features_per_level": enc.enc_xy.n_features_per_level,
                    "log2_hashmap_size": enc.enc_xy.log2_hashmap_size,
                    "base_resolution": float(enc.enc_xy.base_resolution.cpu()),
                    "combine": enc.combine,
                    "kronecker_encoder_state": enc.state_dict(),
                })
            elif self.encoding_type.startswith("DUAL_") and self.dual_encoder is not None:
                enc = self.dual_encoder
                dic.update({
                    "bounding_box": (
                        enc.enc_low.bounding_box
                        if hasattr(enc.enc_low, "bounding_box")
                        else None
                    ),
                    "n_levels_low": (
                        enc.enc_low.n_levels
                        if hasattr(enc.enc_low, "n_levels")
                        else None
                    ),
                    "n_levels_high": (
                        enc.enc_high.n_levels
                        if hasattr(enc.enc_high, "n_levels")
                        else None
                    ),
                    "n_features_per_level": (
                        enc.enc_low.n_features_per_level
                        if hasattr(enc.enc_low, "n_features_per_level")
                        else None
                    ),
                    "log2_hashmap_size": (
                        enc.enc_low.log2_hashmap_size
                        if hasattr(enc.enc_low, "log2_hashmap_size")
                        else None
                    ),
                    "base_resolution_low": (
                        float(enc.enc_low.base_resolution.cpu())
                        if hasattr(enc.enc_low, "base_resolution")
                        else None
                    ),
                    "finest_resolution_low": (
                        float(enc.enc_low.finest_resolution.cpu())
                        if hasattr(enc.enc_low, "finest_resolution")
                        else None
                    ),
                    "base_resolution_high": (
                        float(enc.enc_high.base_resolution.cpu())
                        if hasattr(enc.enc_high, "base_resolution")
                        else None
                    ),
                    "finest_resolution_high": (
                        float(enc.enc_high.finest_resolution.cpu())
                        if hasattr(enc.enc_high, "finest_resolution")
                        else None
                    ),
                    "sigma_low": getattr(enc.enc_low, "sigma", None),
                    "sigma_high": getattr(enc.enc_high, "sigma", None),
                    "n_freq": getattr(enc.enc_low, "n_freq", None),
                    "use_gate": enc.use_gate,
                    "hf_activate_ratio": enc.hf_activate_ratio,
                    "hf_max_weight": enc.hf_max_weight,
                    "dual_encoder_state": enc.state_dict(),
                })

        return dic
