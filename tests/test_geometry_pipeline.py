"""Shared ETKDG geometry and model-contract regressions."""

from pathlib import Path

import torch

from molgnn.featurizer import featurize_smiles
from molgnn.geometry import ETKDG_V3_SEED, ensure_sample_geometry
from molgnn.models.registration import register_builtin_models
from molgnn.registry import get_model_spec
from molgnn.transforms import get_graph_transform, register_builtin_transforms


def _sample(smiles: str = "C1CCCCC1"):
    return featurize_smiles(
        smiles,
        targets=[0.0],
        target_mask=[True],
        sample_id=4,
    )


def test_etkdg_v3_geometry_is_deterministic_and_nonplanar() -> None:
    first, first_record = ensure_sample_geometry(_sample())
    second, second_record = ensure_sample_geometry(_sample())

    assert ETKDG_V3_SEED == 0x5C4E
    assert first_record.source == "etkdg_v3"
    assert first_record.coordinate_sha256 == second_record.coordinate_sha256
    assert torch.equal(first.pos, second.pos)
    centered = first.pos - first.pos.mean(dim=0, keepdim=True)
    assert torch.linalg.matrix_rank(centered) == 3
    assert first.geometry_is_proxy.tolist() == [True]


def test_native_geometry_is_preserved() -> None:
    sample = _sample("CCO")
    native_pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.1, 0.2], [2.0, 1.1, -0.3]],
        dtype=torch.float32,
    )
    sample.pos = native_pos
    sample.atomic_number = torch.tensor([6, 6, 8], dtype=torch.long)

    enriched, record = ensure_sample_geometry(sample)

    assert record.source == "native"
    assert torch.equal(enriched.pos, native_pos)
    assert enriched.geometry_is_proxy.tolist() == [False]


def test_coordinate_model_transforms_consume_shared_geometry() -> None:
    register_builtin_models()
    register_builtin_transforms()
    sample, _ = ensure_sample_geometry(_sample())

    expected_fields = {
        "dimenet": "dimenet_triplet_edge_index",
        "mpnn_3d_distance_bins": "mpnn_3d_edge_type",
        "fragnet": "edge_index_bonds_graph",
    }
    for model_name, field in expected_fields.items():
        spec = get_model_spec(model_name)
        transform = get_graph_transform(spec.graph_transform_name)
        assert transform is not None
        transformed = transform(sample)
        assert torch.equal(transformed.pos, sample.pos)
        assert isinstance(getattr(transformed, field), torch.Tensor)


def test_builtin_geometry_contracts_and_single_generator_location() -> None:
    register_builtin_models()
    required = {
        "dimenet",
        "egnn",
        "eqgat",
        "equiformer",
        "ewaldmp",
        "fragnet",
        "gemnet_q",
        "gemnet_t",
        "hmgnn",
        "mat",
        "mpnn_3d_distance_bins",
        "painn",
        "schnet",
        "visnet",
    }
    assert {
        name
        for name in required
        if get_model_spec(name).geometry_requirement == "required"
    } == required
    assert get_model_spec("potentialnet").geometry_requirement == "optional"
    assert get_model_spec("potentialnet").geometry_role == "hybrid"

    transform_root = Path(__file__).parents[1] / "src" / "molgnn" / "transforms"
    for path in transform_root.glob("*.py"):
        assert "EmbedMolecule" not in path.read_text(encoding="utf-8")
