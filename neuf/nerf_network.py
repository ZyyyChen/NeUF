"""超声神经场的网络结构、编码器初始化、查询接口与检查点序列化。

本模块只保留当前实验仍在使用的三种固定几何模型：基础单头 HashGrid、
无门控双 HashGrid 单头模型，以及显式分离解剖结构与散斑的双头模型。已经移除的
视角方向编码、不确定性预测、位姿分支和射线物理分支不会在这里静默恢复。

除非单个函数另有说明，输入坐标张量最后一维均为三维空间坐标，前面的维度可以是
任意采样布局；查询时会临时展平为 ``[点数, 3]``，输出强度的最后一维为 1。
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuf.dual_freq_encoder import DualFreqEncoder
from neuf.hash_encoder import HashEncoder


# 模型统一放置到当前可用的 CUDA 设备；没有 CUDA 时自动回退到 CPU。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SkipMLP(nn.Module):
    """带一次可选输入跳跃连接的 ReLU 多层感知机。

    跳跃连接发生在指定隐藏层的线性变换之前：把原始输入特征与上一层隐藏特征
    沿最后一维拼接，再送入该层。这样既保留高频编码中的原始信息，也避免深层
    MLP 完全依赖连续非线性变换后的表示。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        *,
        skip_before: int | None = None,
        output_dim: int = 1,
    ) -> None:
        """构建解码器。

        参数:
            input_dim: 每个采样点的输入特征维数。
            hidden_dim: 每个隐藏层的输出维数。
            hidden_layers: 隐藏层数量。
            skip_before: 在第几个隐藏层之前拼接原始输入；``None`` 表示不使用
                跳跃连接，索引从 0 开始。
            output_dim: 最终线性输出层的通道数。
        """
        super().__init__()
        # 显式保存结构参数，便于检查点记录、调试和参数量核对。
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.skip_before = skip_before

        # 逐层计算实际输入维数；跳跃层需要额外容纳一份原始输入特征。
        layers = []
        for index in range(self.hidden_layers):
            layer_input = self.input_dim if index == 0 else self.hidden_dim
            if index == self.skip_before:
                layer_input += self.input_dim
            layers.append(nn.Linear(layer_input, self.hidden_dim))
        self.layers = nn.ModuleList(layers)
        # 输出层不使用激活函数，具体的强度约束由上层查询逻辑决定。
        self.output = nn.Linear(self.hidden_dim, int(output_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """把编码特征解码为每点预测值。

        ``features`` 的形状为 ``[..., input_dim]``，返回形状为
        ``[..., output_dim]``；除最后一维外的所有批次维度都会原样保留。
        """
        hidden = features
        for index, layer in enumerate(self.layers):
            if index == self.skip_before:
                # 仅在指定位置拼接一次，拼接方向始终为特征维（最后一维）。
                hidden = torch.cat((hidden, features), dim=-1)
            hidden = F.relu(layer(hidden))
        return self.output(hidden)


class NeRF(nn.Module):
    """只包含当前保留的 HashGrid 方案的神经超声场。

    三种 ``field_head`` 的用途如下：

    * ``legacy_fixed_geometry``：基础单头 HashGrid，对每个坐标直接预测一个强度；
    * ``dual_single_head_matched``：直接拼接低频与高频特征后，用与基础模型相同
      的单头解码器，作为只替换编码器的公平对照；
    * ``anatomy_speckle_v1``：低频分支预测解剖基底，解剖输出与高频特征共同
      预测散斑残差，
      最终通过 ``anatomy + alpha * speckle`` 合成强度。

    本类不会创建已经淘汰的方向、不确定性、位姿、矢状面、Fourier、Kronecker
    或射线物理分支。遇到旧检查点时会明确拒绝不受支持的结构，避免加载成功但
    实际模型语义不一致。
    """

    # 这些字符串会写入检查点，是模型结构协议的一部分，不能随意改名。
    LEGACY_FIELD_HEAD = "legacy_fixed_geometry"
    MATCHED_FIELD_HEAD = "dual_single_head_matched"
    ANATOMY_SPECKLE_FIELD_HEAD = "anatomy_speckle_v1"

    # 每种场头只允许一种编码器，避免把 E0 的“基础单 Hash”对照静默运行成
    # 双频编码。该映射同时用于新模型初始化和检查点恢复。
    FIELD_HEAD_ENCODINGS = {
        LEGACY_FIELD_HEAD: "HASH",
        MATCHED_FIELD_HEAD: "DUAL_HASH",
        ANATOMY_SPECKLE_FIELD_HEAD: "DUAL_HASH",
    }

    # 非 legacy 模型通过版本号和结构字典共同验证检查点兼容性。
    MODEL_SCHEMA_VERSION = 1

    # 该字典不仅用于说明结构，也会原样写入检查点并在恢复时严格比较。
    # E1 固定为 E0 同款解码器；E2 固定双头结构，避免实验配置静默漂移。
    FIELD_HEAD_CONFIGS = {
        MATCHED_FIELD_HEAD: {
            # E1 只替换 E0 的编码器，解码器结构保持完全相同。
            "input": "combined_raw",
            "hidden_layers": 8,
            "hidden_width": 256,
            "skip_before_hidden_index": 5,
            "output_channels": 1,
            "use_gate": False,
            "progressive_high_frequency": False,
        },
        ANATOMY_SPECKLE_FIELD_HEAD: {
            # A 只由低频特征生成；S 同时读取 A 和原始高频特征。
            "anatomy_input": "feat_low",
            "anatomy_hidden_layers": 4,
            "anatomy_hidden_width": 128,
            "anatomy_skip_before_hidden_index": 2,
            "speckle_input": "concat(anatomy, feat_high)",
            "speckle_hidden_layers": 3,
            "speckle_hidden_width": 64,
            "use_gate": False,
            "progressive_high_frequency": False,
            "training": "joint",
            "loss": "masked_mse",
        },
    }

    def __init__(
        self,
        ckpt: dict | None = None,
        intensity_activation: str | None = None,
        field_head: str | None = None,
        default_alpha: float | None = None,
    ) -> None:
        """创建空模型，或从检查点完整恢复模型。

        参数:
            ckpt: ``get_save_dict`` 产生的检查点字典。传入时会先根据元数据重建
                编码器和解码器，再加载参数。
            intensity_activation: legacy 分支的输出激活，可为 ``identity`` 或
                ``sigmoid``；省略时优先沿用检查点设置。
            field_head: 要使用的场解码结构。恢复检查点时必须与其中记录的结构
                一致，避免把一组权重加载进含义不同的网络。
            default_alpha: 解剖/散斑合成时的默认散斑比例，必须位于 [0, 1]；
                单次 ``query`` 可以用同名参数覆盖它。
        """
        super().__init__()

        # 新模型默认使用基础单头；旧检查点没有 field_head 时也按 legacy 解释。
        checkpoint_head = (
            str(ckpt.get("field_head", self.LEGACY_FIELD_HEAD)).lower()
            if ckpt is not None
            else self.LEGACY_FIELD_HEAD
        )
        self.field_head = checkpoint_head if field_head is None else str(field_head).lower()
        valid_heads = {
            self.LEGACY_FIELD_HEAD,
            self.MATCHED_FIELD_HEAD,
            self.ANATOMY_SPECKLE_FIELD_HEAD,
        }
        # 尽早拒绝拼写错误或已经删除的网络头，防止后面出现难理解的空模块错误。
        if self.field_head not in valid_heads:
            raise ValueError(
                f"Unknown field_head={self.field_head!r}; expected {sorted(valid_heads)}"
            )
        # 显式传入的结构不能覆盖检查点自己的结构协议。
        if ckpt is not None and self.field_head != checkpoint_head:
            raise ValueError(
                "Requested field head does not match checkpoint: "
                f"requested={self.field_head}, checkpoint={checkpoint_head}"
            )

        # alpha 和输出激活都采用“调用参数优先、检查点次之、默认值最后”的规则。
        checkpoint_alpha = 1.0 if ckpt is None else float(ckpt.get("default_alpha", 1.0))
        self.default_alpha = self._validate_alpha(
            checkpoint_alpha if default_alpha is None else default_alpha
        )
        activation = (
            ckpt.get("intensity_activation", "identity")
            if ckpt is not None and intensity_activation is None
            else ("identity" if intensity_activation is None else intensity_activation)
        )
        self.intensity_activation = str(activation).lower()
        if self.intensity_activation not in {"identity", "sigmoid"}:
            raise ValueError("intensity_activation must be 'identity' or 'sigmoid'")

        # 以下字段记录尚未初始化或训练过程中会变化的模型状态。
        self.encoding_type = ""
        self.encoding_initialized = False
        self.use_encoding = True
        # 当前保留的三种模型均与观察方向无关；字段仅为旧调用接口兼容而保留。
        self.use_direction = False
        self.in_ch = 0
        self.out_ch = 1
        self.network_depth = 8
        self.network_width = 256
        self.training_progress = 0.0

        # 编码器和解码头按所选结构延迟创建。Optional 类型让“尚未初始化”状态
        # 显式可见；使用前必须检查非空，不能假定所有分支都会同时存在。
        self.hash_encoder: HashEncoder | None = None
        self.dual_encoder: DualFreqEncoder | None = None
        self.legacy_head: SkipMLP | None = None
        self.matched_head: SkipMLP | None = None
        self.anatomy_head: SkipMLP | None = None
        self.speckle_head: SkipMLP | None = None

        if ckpt is not None:
            # 恢复过程会依次建立编码器、解码器并加载 state_dict。
            self._init_from_ckpt(ckpt)

    def _init_from_ckpt(self, ckpt: dict) -> None:
        if str(ckpt.get("output_mode", "intensity")).lower() != "intensity":
            raise ValueError("Only retained point-intensity checkpoints are supported")
        if bool(ckpt.get("use_directions", False)):
            raise ValueError("View-dependent checkpoints belong to a removed model variant")

        encoding = str(ckpt.get("encoding", "")).upper()
        expected_encoding = self.FIELD_HEAD_ENCODINGS[self.field_head]
        if encoding != expected_encoding:
            raise ValueError(
                f"{self.field_head} requires encoding={expected_encoding!r}, got {encoding!r}"
            )
        if self.field_head != self.LEGACY_FIELD_HEAD:
            if int(ckpt.get("model_schema_version", -1)) != self.MODEL_SCHEMA_VERSION:
                raise ValueError("Missing or unsupported Phase 1 model schema")
            if ckpt.get("field_head_config") != self.FIELD_HEAD_CONFIGS[self.field_head]:
                raise ValueError(f"Incompatible field_head_config for {self.field_head}")

        if encoding == "HASH":
            self.init_hash_encoding(
                bounding_box=ckpt["bounding_box"],
                n_levels=int(ckpt.get("n_levels", 16)),
                n_features_per_level=int(ckpt.get("n_features_per_level", 2)),
                log2_hashmap_size=int(ckpt.get("log2_hashmap_size", 19)),
                base_resolution=int(float(ckpt.get("base_resolution", 16))),
                finest_resolution=int(float(ckpt.get("finest_resolution", 512))),
            )
        elif encoding == "DUAL_HASH":
            self.init_dual_encoding(
                bounding_box=ckpt["bounding_box"],
                n_levels_low=int(ckpt.get("n_levels_low", 8)),
                n_levels_high=int(ckpt.get("n_levels_high", 8)),
                n_features_per_level=int(ckpt.get("n_features_per_level", 2)),
                log2_hashmap_size=int(ckpt.get("log2_hashmap_size", 19)),
                base_resolution_low=int(float(ckpt.get("base_resolution_low", 16))),
                finest_resolution_low=int(float(ckpt.get("finest_resolution_low", 128))),
                base_resolution_high=int(float(ckpt.get("base_resolution_high", 64))),
                finest_resolution_high=int(float(ckpt.get("finest_resolution_high", 512))),
                use_gate=bool(ckpt.get("use_gate", True)),
                hf_activate_ratio=float(ckpt.get("hf_activate_ratio", 0.2)),
                hf_max_weight=float(ckpt.get("hf_max_weight", 1.0)),
            )
        else:
            raise ValueError(
                f"Checkpoint encoding {encoding!r} was removed; use HASH or DUAL_HASH"
            )

        self.init_model(
            D=int(ckpt.get("network_depth", 8)),
            W=int(ckpt.get("network_width", 256)),
        )
        state = ckpt.get("network_fn_state_dict")
        if state is None:
            raise KeyError("Checkpoint is missing 'network_fn_state_dict'")
        if self.field_head == self.LEGACY_FIELD_HEAD:
            state = self._translate_legacy_state(state)
        try:
            self.load_state_dict(state, strict=self.field_head != self.LEGACY_FIELD_HEAD)
        except RuntimeError as error:
            raise ValueError(f"Incompatible {self.field_head} checkpoint state") from error
        self.training_progress = float(ckpt.get("training_progress", 1.0))

    @staticmethod
    def _translate_legacy_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Map the retained old single-head checkpoint keys to the compact model.

        旧网络在 skip 层按 ``[原始特征, 隐藏特征]`` 拼接，当前 ``SkipMLP``
        按 ``[隐藏特征, 原始特征]`` 拼接。两种写法的权重形状相同，因此仅改键名
        会静默加载但改变输出；这里同时重排旧 skip 权重的输入列。
        """
        if any(key.startswith("legacy_head.") for key in state):
            return state
        translated = {}
        for key, value in state.items():
            if key.startswith("pts_linears."):
                parts = key.split(".")
                layer_index = int(parts[1])
                if (
                    parts[2] == "weight"
                    and layer_index > 0
                    and value.ndim == 2
                    and value.shape[1] > value.shape[0]
                ):
                    input_dim = value.shape[1] - value.shape[0]
                    value = torch.cat(
                        (value[:, input_dim:], value[:, :input_dim]),
                        dim=1,
                    )
                key = "legacy_head.layers." + key.removeprefix("pts_linears.")
            elif key.startswith("output_linear."):
                key = "legacy_head.output." + key.removeprefix("output_linear.")
            elif key.startswith("encode."):
                key = "hash_encoder." + key.removeprefix("encode.")
            elif key.startswith(("sigma_linear.", "views_linears.", "feature_linear.")):
                continue
            translated[key] = value
        return translated

    def _require_uninitialized(self) -> None:
        if self.encoding_initialized:
            raise RuntimeError("Encoding has already been initialized")

    def init_hash_encoding(
        self,
        bounding_box,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 256,
        **legacy_kwargs,
    ) -> None:
        self._require_uninitialized()
        if self.FIELD_HEAD_ENCODINGS[self.field_head] != "HASH":
            raise ValueError(f"{self.field_head} requires DUAL_HASH")
        if bool(legacy_kwargs.get("use_directions", False)):
            raise ValueError("The retained basic HashGrid is direction independent")
        self.hash_encoder = HashEncoder(
            bounding_box,
            n_levels,
            n_features_per_level,
            log2_hashmap_size,
            base_resolution,
            finest_resolution,
        )
        self.in_ch = self.hash_encoder.out_dim
        self.encoding_type = "HASH"
        self.encoding_initialized = True

    def init_dual_encoding(
        self,
        *,
        bounding_box,
        n_levels_low: int = 8,
        n_levels_high: int = 8,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution_low: int = 16,
        finest_resolution_low: int = 64,
        base_resolution_high: int = 64,
        finest_resolution_high: int = 512,
        use_gate: bool = False,
        hf_activate_ratio: float = 0.2,
        hf_max_weight: float = 1.0,
        pe_type: str = "hash",
        **removed_kwargs,
    ) -> None:
        self._require_uninitialized()
        if self.FIELD_HEAD_ENCODINGS[self.field_head] != "DUAL_HASH":
            raise ValueError(f"{self.field_head} requires HASH")
        if str(pe_type).lower() != "hash":
            raise ValueError("Only the retained DUAL_HASH encoder is supported")
        self.dual_encoder = DualFreqEncoder(
            bounding_box=bounding_box,
            n_levels_low=n_levels_low,
            n_levels_high=n_levels_high,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution_low=base_resolution_low,
            finest_resolution_low=finest_resolution_low,
            base_resolution_high=base_resolution_high,
            finest_resolution_high=finest_resolution_high,
            use_gate=False,
            hf_activate_ratio=hf_activate_ratio,
            hf_max_weight=hf_max_weight,
        )
        self.in_ch = self.dual_encoder.out_dim
        self.encoding_type = "DUAL_HASH"
        self.encoding_initialized = True

    def init_model(self, D: int = 8, W: int = 256) -> None:
        if not self.encoding_initialized:
            raise RuntimeError("Initialize HASH or DUAL_HASH before the decoder")
        self.network_depth = int(D)
        self.network_width = int(W)

        if self.field_head == self.LEGACY_FIELD_HEAD:
            self.legacy_head = SkipMLP(
                self.in_ch,
                self.network_width,
                self.network_depth,
                skip_before=5 if self.network_depth > 5 else None,
            )
        else:
            if self.encoding_type != "DUAL_HASH" or self.dual_encoder is None:
                raise ValueError(f"{self.field_head} requires DUAL_HASH")
            if self.field_head == self.MATCHED_FIELD_HEAD:
                self.matched_head = SkipMLP(
                    self.dual_encoder.out_dim,
                    self.network_width,
                    self.network_depth,
                    skip_before=5 if self.network_depth > 5 else None,
                )
            else:
                self.anatomy_head = SkipMLP(
                    self.dual_encoder.out_dim_low,
                    128,
                    4,
                    skip_before=2,
                )
                self.speckle_head = SkipMLP(
                    self.dual_encoder.out_dim_high + 1,
                    64,
                    3,
                )
            self._validate_phase1_decoder_size()
        self.to(DEVICE)

    @staticmethod
    def _validate_alpha(alpha: float) -> float:
        value = float(alpha)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"alpha must be finite and in [0, 1], got {value}")
        return value

    @staticmethod
    def _chunk_apply(
        inputs: torch.Tensor,
        chunk: int | None,
        function: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if chunk is None:
            return function(inputs)
        return torch.cat(
            [function(inputs[index:index + chunk]) for index in range(0, inputs.shape[0], chunk)],
            dim=0,
        )

    def _encoded(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.encoding_type == "HASH" and self.hash_encoder is not None:
            return self.hash_encoder(inputs)
        if self.encoding_type == "DUAL_HASH" and self.dual_encoder is not None:
            return self.dual_encoder(inputs, self.training_progress)
        raise RuntimeError("Hash encoder is not initialized")

    def query_components(
        self,
        inputs: torch.Tensor,
        dirs: torch.Tensor | None = None,
        netchunk: int | None = 1024 * 64,
        *,
        alpha: float | None = None,
    ) -> dict[str, torch.Tensor]:
        del dirs
        if self.field_head != self.ANATOMY_SPECKLE_FIELD_HEAD:
            raise ValueError("Components are available only for anatomy_speckle_v1")
        dual_encoder = self.dual_encoder
        anatomy_head = self.anatomy_head
        speckle_head = self.speckle_head
        if dual_encoder is None or anatomy_head is None or speckle_head is None:
            raise RuntimeError("Anatomy/speckle model is not initialized")
        alpha_value = self._validate_alpha(self.default_alpha if alpha is None else alpha)
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])

        def query_chunk(coordinates: torch.Tensor) -> torch.Tensor:
            features = dual_encoder.forward_decomposed(
                coordinates,
                self.training_progress,
            )
            anatomy = anatomy_head(features["feat_low"])
            speckle_input = torch.cat((anatomy, features["feat_high"]), dim=-1)
            speckle = speckle_head(speckle_input)
            return torch.cat((anatomy, speckle), dim=-1)

        values = self._chunk_apply(flat_inputs, netchunk, query_chunk)
        anatomy, speckle = values[:, :1], values[:, 1:2]
        return {
            "anatomy": anatomy,
            "speckle": speckle,
            "intensity": anatomy + alpha_value * speckle,
        }

    def query(
        self,
        inputs: torch.Tensor,
        dirs: torch.Tensor | None = None,
        netchunk: int | None = 1024 * 64,
        *,
        alpha: float | None = None,
        **removed_kwargs,
    ) -> torch.Tensor:
        del dirs
        if removed_kwargs.get("return_sigma", False):
            raise ValueError("The removed uncertainty head is no longer available")
        if self.field_head == self.ANATOMY_SPECKLE_FIELD_HEAD:
            return self.query_components(inputs, netchunk=netchunk, alpha=alpha)["intensity"]

        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        if self.field_head == self.MATCHED_FIELD_HEAD:
            dual_encoder = self.dual_encoder
            matched_head = self.matched_head
            if dual_encoder is None or matched_head is None:
                raise RuntimeError("Matched single-head model is not initialized")

            def matched(coordinates: torch.Tensor) -> torch.Tensor:
                features = dual_encoder.forward_decomposed(
                    coordinates,
                    self.training_progress,
                )
                return matched_head(features["combined_raw"])

            return self._chunk_apply(flat_inputs, netchunk, matched)

        legacy_head = self.legacy_head
        if legacy_head is None:
            raise RuntimeError("Basic single-head model is not initialized")

        def legacy(coordinates: torch.Tensor) -> torch.Tensor:
            prediction = legacy_head(self._encoded(coordinates))
            if self.intensity_activation == "sigmoid":
                prediction = torch.sigmoid(prediction)
            return prediction

        return self._chunk_apply(flat_inputs, netchunk, legacy)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.query(inputs)

    def grad_vars(self) -> list[nn.Parameter]:
        return list(self.parameters())

    def decoder_parameter_count(self) -> int:
        heads = {
            self.LEGACY_FIELD_HEAD: (self.legacy_head,),
            self.MATCHED_FIELD_HEAD: (self.matched_head,),
            self.ANATOMY_SPECKLE_FIELD_HEAD: (self.anatomy_head, self.speckle_head),
        }[self.field_head]
        return sum(
            parameter.numel()
            for head in heads
            if head is not None
            for parameter in head.parameters()
        )

    def parameter_counts(self) -> dict[str, int]:
        encoder = self.hash_encoder if self.hash_encoder is not None else self.dual_encoder
        encoder_count = 0 if encoder is None else sum(p.numel() for p in encoder.parameters())
        return {
            "encoder": int(encoder_count),
            "decoder": int(self.decoder_parameter_count()),
            "total": int(sum(parameter.numel() for parameter in self.parameters())),
        }

    def _validate_phase1_decoder_size(self) -> None:
        expected = {
            self.ANATOMY_SPECKLE_FIELD_HEAD: 63426,
            self.MATCHED_FIELD_HEAD: 477441,
        }
        actual = self.decoder_parameter_count()
        if actual != expected[self.field_head]:
            raise RuntimeError(
                f"Unexpected {self.field_head} decoder size: "
                f"expected={expected[self.field_head]}, actual={actual}"
            )

    def set_phase1_training_stage(self, progress: float) -> int:
        stage = 1 if progress < 0.2 else (2 if progress < 0.8 else 3)
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        return stage

    def get_encode_name(self) -> str:
        return self.encoding_type

    def get_rep_name(self) -> str:
        return self.encoding_type

    def get_save_dict(self) -> dict:
        payload = {
            "encoding": self.encoding_type,
            "use_directions": False,
            "use_encoding": True,
            "intensity_activation": self.intensity_activation,
            "network_depth": self.network_depth,
            "network_width": self.network_width,
            "field_head": self.field_head,
            "model_schema_version": (
                self.MODEL_SCHEMA_VERSION
                if self.field_head != self.LEGACY_FIELD_HEAD
                else None
            ),
            "default_alpha": self.default_alpha,
            "field_head_config": self.FIELD_HEAD_CONFIGS.get(self.field_head),
            "training_progress": float(self.training_progress),
            "parameter_counts": self.parameter_counts(),
            "network_fn_state_dict": self.state_dict(),
        }
        if self.hash_encoder is not None:
            encoder = self.hash_encoder
            payload.update(
                {
                    "bounding_box": encoder.bounding_box,
                    "n_levels": encoder.n_levels,
                    "n_features_per_level": encoder.n_features_per_level,
                    "log2_hashmap_size": encoder.log2_hashmap_size,
                    "base_resolution": float(encoder.base_resolution.cpu()),
                    "finest_resolution": float(encoder.finest_resolution.cpu()),
                }
            )
        elif self.dual_encoder is not None:
            encoder = self.dual_encoder
            payload.update(
                {
                    "bounding_box": encoder.enc_low.bounding_box,
                    "n_levels_low": encoder.enc_low.n_levels,
                    "n_levels_high": encoder.enc_high.n_levels,
                    "n_features_per_level": encoder.enc_low.n_features_per_level,
                    "log2_hashmap_size": encoder.enc_low.log2_hashmap_size,
                    "base_resolution_low": float(encoder.enc_low.base_resolution.cpu()),
                    "finest_resolution_low": float(encoder.enc_low.finest_resolution.cpu()),
                    "base_resolution_high": float(encoder.enc_high.base_resolution.cpu()),
                    "finest_resolution_high": float(encoder.enc_high.finest_resolution.cpu()),
                    "use_gate": encoder.use_gate,
                    "hf_activate_ratio": encoder.hf_activate_ratio,
                    "hf_max_weight": encoder.hf_max_weight,
                }
            )
        return payload
