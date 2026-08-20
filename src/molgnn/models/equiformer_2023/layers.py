"""Equivariant attention primitives for the 2023 Equiformer core.

This module intentionally uses only public e3nn APIs so it can replace the
author source's e3nn-0.4 private helpers while retaining its tensor-product
topology, scalar-only attention scores, and fan-in rescaling convention.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from e3nn import nn as e3nn_nn
from e3nn import o3
from torch import Tensor, nn
from torch.nn import functional
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import scatter, softmax

from .radial import RadialProfile


class TensorProductRescale(nn.Module):
    """Tensor product with source-style fan-in scaling and scalar-only bias."""

    def __init__(
        self,
        irreps_in1: o3.Irreps | str,
        irreps_in2: o3.Irreps | str,
        irreps_out: o3.Irreps | str,
        instructions: list[tuple[int, int, int, str, bool]],
        *,
        bias: bool,
        internal_weights: bool,
        shared_weights: bool,
    ) -> None:
        super().__init__()
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        self.tp = o3.TensorProduct(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            instructions,
            irrep_normalization="component",
            path_normalization="none",
            internal_weights=internal_weights,
            shared_weights=shared_weights,
        )
        scales = self._path_scales()
        self.register_buffer("radial_output_scale", self._weight_scales(scales))
        if self.tp.internal_weights:
            with torch.no_grad():
                for instruction_index, _, view in self.tp.weight_views(
                    yield_instruction=True
                ):
                    view.mul_(scales[instruction_index])

        self._bias_slices: list[slice] = []
        bias_parameters: list[nn.Parameter] = []
        if bias:
            for out_slice, (multiplicity, irrep) in zip(
                self.irreps_out.slices(), self.irreps_out, strict=True
            ):
                if irrep.l == 0 and irrep.p == 1:
                    self._bias_slices.append(out_slice)
                    bias_parameters.append(nn.Parameter(torch.zeros(multiplicity)))
        self.bias = nn.ParameterList(bias_parameters)

    def _path_scales(self) -> tuple[float, ...]:
        fan_in: dict[int, int] = {}
        for instruction in self.tp.instructions:
            if instruction.connection_mode == "uvw":
                value = (
                    self.irreps_in1[instruction.i_in1].mul
                    * self.irreps_in2[instruction.i_in2].mul
                )
            elif instruction.connection_mode == "uvu":
                value = self.irreps_in2[instruction.i_in2].mul
            else:  # pragma: no cover - this core creates only uvw and uvu paths
                raise ValueError(
                    f"unsupported Equiformer tensor-product mode {instruction.connection_mode}"
                )
            fan_in[instruction.i_out] = fan_in.get(instruction.i_out, 0) + value
        return tuple(
            1.0 / math.sqrt(fan_in[instruction.i_out])
            for instruction in self.tp.instructions
        )

    def _weight_scales(self, path_scales: Sequence[float]) -> Tensor:
        pieces: list[Tensor] = []
        for instruction, scale in zip(self.tp.instructions, path_scales, strict=True):
            if instruction.has_weight:
                width = math.prod(instruction.path_shape)
                pieces.append(torch.full((width,), scale, dtype=torch.float32))
        if not pieces:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(pieces)

    def forward(self, first: Tensor, second: Tensor, weight: Tensor | None = None) -> Tensor:
        if self.tp.internal_weights:
            if weight is not None:
                raise ValueError("internal tensor products do not accept external weights")
            output = self.tp(first, second)
        else:
            if weight is None:
                raise ValueError("distance-conditioned tensor product requires external weights")
            expected = self.tp.weight_numel
            if weight.shape != (first.shape[0], expected):
                raise ValueError(
                    "external tensor-product weights must have shape "
                    f"[{first.shape[0]}, {expected}]"
                )
            output = self.tp(first, second, weight)
        if not self.bias:
            return output
        output = output.clone()
        for out_slice, bias in zip(self._bias_slices, self.bias, strict=True):
            output[:, out_slice] = output[:, out_slice] + bias
        return output


class EquivariantLinear(nn.Module):
    """The source's LinearRS: a tensor product with one invariant scalar."""

    def __init__(
        self,
        irreps_in: o3.Irreps | str,
        irreps_out: o3.Irreps | str,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        scalar = o3.Irreps("1x0e")
        instructions = _fully_connected_instructions(self.irreps_in, scalar, self.irreps_out)
        self.tp = TensorProductRescale(
            self.irreps_in,
            scalar,
            self.irreps_out,
            instructions,
            bias=bias,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(self, features: Tensor) -> Tensor:
        scalar = features.new_ones((features.shape[0], 1))
        return self.tp(features, scalar)


class EquivariantLayerNorm(nn.Module):
    """Per-node LayerNormV2 from the author implementation."""

    def __init__(self, irreps: o3.Irreps | str, eps: float = 1e-5) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.irreps.num_irreps))
        scalar_count = sum(
            multiplicity
            for multiplicity, irrep in self.irreps
            if irrep.l == 0 and irrep.p == 1
        )
        self.bias = nn.Parameter(torch.zeros(scalar_count))

    def forward(self, features: Tensor) -> Tensor:
        fields: list[Tensor] = []
        feature_offset = 0
        weight_offset = 0
        bias_offset = 0
        for multiplicity, irrep in self.irreps:
            width = multiplicity * irrep.dim
            field = features[:, feature_offset : feature_offset + width].reshape(
                -1, multiplicity, irrep.dim
            )
            feature_offset += width
            if irrep.l == 0 and irrep.p == 1:
                field = field - field.mean(dim=1, keepdim=True)
            norm = field.square().mean(dim=-1).mean(dim=1, keepdim=True)
            scale = (norm + self.eps).rsqrt()
            scale = scale * self.weight[weight_offset : weight_offset + multiplicity]
            weight_offset += multiplicity
            field = field * scale.reshape(-1, multiplicity, 1)
            if irrep.l == 0 and irrep.p == 1:
                field = field + self.bias[bias_offset : bias_offset + multiplicity].reshape(
                    1, multiplicity, 1
                )
                bias_offset += multiplicity
            fields.append(field.reshape(-1, width))
        return torch.cat(fields, dim=-1)


class SmoothLeakyReLU(nn.Module):
    """Smooth LeakyReLU used for Equiformer's scalar attention logits."""

    def __init__(self, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, features: Tensor) -> Tensor:
        alpha = self.negative_slope
        return ((1.0 + alpha) / 2.0) * features + ((1.0 - alpha) / 2.0) * features * (
            2.0 * torch.sigmoid(features) - 1.0
        )


class SeparableFCTP(nn.Module):
    """Depthwise geometric tensor product then equivariant linear projection."""

    def __init__(
        self,
        irreps_input: o3.Irreps | str,
        irreps_edge: o3.Irreps | str,
        irreps_output: o3.Irreps | str,
        *,
        radial_channels: Sequence[int] | None,
        use_gate: bool,
        internal_weights: bool,
    ) -> None:
        super().__init__()
        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        self.dtp = _depthwise_tensor_product(
            self.irreps_input,
            o3.Irreps(irreps_edge),
            self.irreps_output,
            internal_weights=internal_weights,
        )
        self.radial: RadialProfile | None = None
        if radial_channels is not None:
            self.radial = RadialProfile(
                [*radial_channels, self.dtp.tp.weight_numel]
            )
            self.radial.apply_output_scale(self.dtp.radial_output_scale)
        if use_gate:
            self.gate = _make_gate(self.irreps_output)
            linear_out = self.gate.irreps_in
        else:
            self.gate = None
            linear_out = self.irreps_output
        self.linear = EquivariantLinear(self.dtp.irreps_out, linear_out)

    def forward(
        self,
        node_features: Tensor,
        edge_attributes: Tensor,
        edge_scalars: Tensor | None,
    ) -> Tensor:
        if self.radial is None:
            raw = self.dtp(node_features, edge_attributes)
        else:
            if edge_scalars is None:
                raise ValueError("distance-conditioned separable tensor product needs radial features")
            raw = self.dtp(node_features, edge_attributes, self.radial(edge_scalars))
        output = self.linear(raw)
        return self.gate(output) if self.gate is not None else output


class EquivariantGraphAttention(nn.Module):
    """Nonlinear-message multihead Equiformer graph attention."""

    def __init__(
        self,
        irreps_input: o3.Irreps | str,
        irreps_edge: o3.Irreps | str,
        irreps_head: o3.Irreps | str,
        *,
        num_heads: int,
        radial_channels: Sequence[int],
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.merge_src = EquivariantLinear(self.irreps_input, self.irreps_input, bias=True)
        self.merge_dst = EquivariantLinear(self.irreps_input, self.irreps_input, bias=False)
        self.irreps_heads = (self.irreps_head * num_heads).sort().irreps.simplify()
        scalar_channels = _scalar_multiplicity(self.irreps_heads)
        if scalar_channels == 0 or scalar_channels % num_heads != 0:
            raise ValueError("Equiformer attention needs equal scalar channels in every head")
        self.scalar_channels_per_head = scalar_channels // num_heads
        self.irreps_alpha = o3.Irreps(f"{scalar_channels}x0e")

        self.message = SeparableFCTP(
            self.irreps_input,
            self.irreps_edge,
            self.irreps_input,
            radial_channels=radial_channels,
            use_gate=True,
            internal_weights=False,
        )
        self.alpha_projection = EquivariantLinear(
            self.message.dtp.irreps_out,
            self.irreps_alpha,
        )
        self.value = SeparableFCTP(
            self.irreps_input,
            self.irreps_edge,
            self.irreps_heads,
            radial_channels=None,
            use_gate=False,
            internal_weights=True,
        )
        self.alpha_to_heads = IrrepsToHeads(self.irreps_alpha, num_heads)
        self.value_to_heads = IrrepsToHeads(self.irreps_heads, num_heads)
        self.heads_to_irreps = HeadsToIrreps(self.irreps_head)
        self.alpha_activation = e3nn_nn.Activation(
            o3.Irreps(f"{self.scalar_channels_per_head}x0e"),
            [SmoothLeakyReLU(0.2)],
        )
        self.alpha_dot = nn.Parameter(
            torch.empty(1, num_heads, self.scalar_channels_per_head)
        )
        glorot(self.alpha_dot)
        self.alpha_dropout = nn.Dropout(attention_dropout)
        self.projection = EquivariantLinear(self.irreps_heads, self.irreps_input)

    def forward(
        self,
        node_features: Tensor,
        source: Tensor,
        target: Tensor,
        edge_attributes: Tensor,
        edge_scalars: Tensor,
    ) -> Tensor:
        node_count = node_features.shape[0]
        if source.numel() == 0:
            return node_features.new_zeros((node_count, self.irreps_input.dim))
        merged = self.merge_src(node_features)[source] + self.merge_dst(node_features)[target]
        assert self.message.radial is not None
        raw = self.message.dtp(merged, edge_attributes, self.message.radial(edge_scalars))
        alpha = self.alpha_to_heads(self.alpha_projection(raw))
        activated = self.message.linear(raw)
        assert self.message.gate is not None
        activated = self.message.gate(activated)
        value = self.value_to_heads(self.value(activated, edge_attributes, None))

        logits = (self.alpha_activation(alpha) * self.alpha_dot).sum(dim=-1)
        weights = softmax(logits, target, num_nodes=node_count).unsqueeze(-1)
        weights = self.alpha_dropout(weights)
        attended = scatter(value * weights, target, dim=0, dim_size=node_count, reduce="sum")
        return self.projection(self.heads_to_irreps(attended))


class EquivariantFeedForward(nn.Module):
    """Two source-style fully connected tensor products with an intervening gate."""

    def __init__(
        self,
        irreps_input: o3.Irreps | str,
        irreps_output: o3.Irreps | str,
        irreps_middle: o3.Irreps | str,
    ) -> None:
        super().__init__()
        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        middle = o3.Irreps(irreps_middle)
        self.gate = _make_gate(middle)
        scalar = o3.Irreps("1x0e")
        self.first = _fully_connected_tensor_product(
            self.irreps_input,
            scalar,
            self.gate.irreps_in,
        )
        self.second = _fully_connected_tensor_product(
            middle,
            scalar,
            self.irreps_output,
        )

    def forward(self, node_features: Tensor, node_attr: Tensor) -> Tensor:
        return self.second(self.gate(self.first(node_features, node_attr)), node_attr)


class EquiformerBlock(nn.Module):
    """Pre-normalized attention and feed-forward residual block."""

    def __init__(
        self,
        irreps_input: o3.Irreps | str,
        irreps_output: o3.Irreps | str,
        irreps_edge: o3.Irreps | str,
        irreps_head: o3.Irreps | str,
        irreps_middle: o3.Irreps | str,
        *,
        num_heads: int,
        radial_channels: Sequence[int],
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        self.norm_attention = EquivariantLayerNorm(self.irreps_input)
        self.attention = EquivariantGraphAttention(
            self.irreps_input,
            irreps_edge,
            irreps_head,
            num_heads=num_heads,
            radial_channels=radial_channels,
            attention_dropout=attention_dropout,
        )
        self.norm_ffn = EquivariantLayerNorm(self.irreps_input)
        self.ffn = EquivariantFeedForward(
            self.irreps_input,
            self.irreps_output,
            irreps_middle,
        )
        self.shortcut = (
            None
            if self.irreps_input == self.irreps_output
            else EquivariantLinear(self.irreps_input, self.irreps_output)
        )

    def forward(
        self,
        node_features: Tensor,
        node_attr: Tensor,
        source: Tensor,
        target: Tensor,
        edge_attributes: Tensor,
        edge_scalars: Tensor,
    ) -> Tensor:
        attended = self.attention(
            self.norm_attention(node_features),
            source,
            target,
            edge_attributes,
            edge_scalars,
        )
        node_features = node_features + attended
        residual = node_features if self.shortcut is None else self.shortcut(node_features)
        return residual + self.ffn(self.norm_ffn(node_features), node_attr)


class EdgeDegreeEmbedding(nn.Module):
    """Source edge-degree embedding normalized by fixed average degree."""

    def __init__(
        self,
        irreps_node: o3.Irreps | str,
        irreps_edge: o3.Irreps | str,
        *,
        radial_channels: Sequence[int],
        average_degree: float,
    ) -> None:
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.average_degree = float(average_degree)
        scalar = o3.Irreps("1x0e")
        self.expand = EquivariantLinear(scalar, self.irreps_node)
        self.dtp = _depthwise_tensor_product(
            self.irreps_node,
            o3.Irreps(irreps_edge),
            self.irreps_node,
            internal_weights=False,
        )
        self.radial = RadialProfile([*radial_channels, self.dtp.tp.weight_numel])
        self.radial.apply_output_scale(self.dtp.radial_output_scale)
        self.project = EquivariantLinear(self.dtp.irreps_out, self.irreps_node)

    def forward(
        self,
        node_features: Tensor,
        source: Tensor,
        target: Tensor,
        edge_attributes: Tensor,
        edge_scalars: Tensor,
    ) -> Tensor:
        node_count = node_features.shape[0]
        if source.numel() == 0:
            return node_features.new_zeros((node_count, self.irreps_node.dim))
        constant = node_features.new_ones((node_count, 1))
        expanded = self.expand(constant)
        edge_features = self.dtp(
            expanded[source],
            edge_attributes,
            self.radial(edge_scalars),
        )
        projected = self.project(edge_features)
        return scatter(projected, target, dim=0, dim_size=node_count, reduce="sum") / math.sqrt(
            self.average_degree
        )


class IrrepsToHeads(nn.Module):
    """Reshape sorted multiplicity blocks into an explicit head axis."""

    def __init__(self, irreps: o3.Irreps | str, num_heads: int) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.num_heads = num_heads
        self._slices = tuple(self.irreps.slices())
        for multiplicity, _ in self.irreps:
            if multiplicity % num_heads != 0:
                raise ValueError("irreps multiplicities must be divisible by num_heads")

    def forward(self, features: Tensor) -> Tensor:
        parts = [
            features[:, value_slice].reshape(features.shape[0], self.num_heads, -1)
            for value_slice in self._slices
        ]
        return torch.cat(parts, dim=-1)


class HeadsToIrreps(nn.Module):
    """Inverse of ``IrrepsToHeads`` for one head's irrep layout."""

    def __init__(self, irreps_head: o3.Irreps | str) -> None:
        super().__init__()
        self.irreps_head = o3.Irreps(irreps_head)
        self._slices = tuple(self.irreps_head.slices())

    def forward(self, features: Tensor) -> Tensor:
        return torch.cat(
            [
                features[:, :, value_slice].reshape(features.shape[0], -1)
                for value_slice in self._slices
            ],
            dim=-1,
        )


def _fully_connected_instructions(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
) -> list[tuple[int, int, int, str, bool]]:
    return [
        (input_one, input_two, output, "uvw", True)
        for input_one, (_, first) in enumerate(irreps_in1)
        for input_two, (_, second) in enumerate(irreps_in2)
        for output, (_, result) in enumerate(irreps_out)
        if result in first * second
    ]


def _fully_connected_tensor_product(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
) -> TensorProductRescale:
    return TensorProductRescale(
        irreps_in1,
        irreps_in2,
        irreps_out,
        _fully_connected_instructions(irreps_in1, irreps_in2, irreps_out),
        bias=True,
        internal_weights=True,
        shared_weights=True,
    )


def _depthwise_tensor_product(
    irreps_node: o3.Irreps,
    irreps_edge: o3.Irreps,
    irreps_requested: o3.Irreps,
    *,
    internal_weights: bool,
) -> TensorProductRescale:
    outputs: list[tuple[int, o3.Irrep]] = []
    instructions: list[tuple[int, int, int, str, bool]] = []
    scalar = o3.Irrep("0e")
    for input_node, (multiplicity, node_irrep) in enumerate(irreps_node):
        for input_edge, (_, edge_irrep) in enumerate(irreps_edge):
            for output_irrep in node_irrep * edge_irrep:
                if output_irrep in irreps_requested or output_irrep == scalar:
                    output_index = len(outputs)
                    outputs.append((multiplicity, output_irrep))
                    instructions.append(
                        (input_node, input_edge, output_index, "uvu", True)
                    )
    if not outputs:  # pragma: no cover - L=0 edge harmonics always create a path
        raise ValueError("Equiformer depthwise tensor product has no allowed paths")
    raw_out = o3.Irreps(outputs)
    ordered = raw_out.sort()
    sorted_instructions = [
        (input_one, input_two, ordered.p[output], mode, trainable)
        for input_one, input_two, output, mode, trainable in instructions
    ]
    return TensorProductRescale(
        irreps_node,
        irreps_edge,
        ordered.irreps,
        sorted_instructions,
        bias=False,
        internal_weights=internal_weights,
        shared_weights=internal_weights,
    )


def _make_gate(irreps: o3.Irreps | str) -> nn.Module:
    desired = o3.Irreps(irreps)
    scalar_blocks: list[tuple[int, o3.Irrep]] = []
    gated_blocks: list[tuple[int, o3.Irrep]] = []
    for multiplicity, irrep in desired:
        if irrep.l == 0 and irrep.p == 1:
            scalar_blocks.append((multiplicity, irrep))
        else:
            gated_blocks.append((multiplicity, irrep))
    scalars = o3.Irreps(scalar_blocks).simplify()
    gated = o3.Irreps(gated_blocks).simplify()
    if gated.dim == 0:
        return e3nn_nn.Activation(scalars, [functional.silu for _ in scalars])
    gates = o3.Irreps([(multiplicity, "0e") for multiplicity, _ in gated]).simplify()
    return e3nn_nn.Gate(
        scalars,
        [functional.silu for _ in scalars],
        gates,
        [torch.sigmoid for _ in gates],
        gated,
    )


def _scalar_multiplicity(irreps: o3.Irreps) -> int:
    return sum(
        multiplicity
        for multiplicity, irrep in irreps
        if irrep.l == 0 and irrep.p == 1
    )


__all__ = [
    "EdgeDegreeEmbedding",
    "EquiformerBlock",
    "EquivariantGraphAttention",
    "EquivariantLayerNorm",
    "EquivariantLinear",
    "SmoothLeakyReLU",
    "TensorProductRescale",
]
