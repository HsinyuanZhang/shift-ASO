"""Pure builder tests on tiny synthetic streams (no ZIP, no real NP data).

All expectations below are hand-derived from the accepted source semantics:
uniform rank-per-parity snippet selection, >=4 combined-snippet validity,
nearest-valid-block fallback with lower-index tie-breaking, unknown -1,
chunk-boundary rank consumption, and the computed-but-not-applied overlap
diagnostic.
"""
import numpy as np
import pytest

from shift_aso.neuropixels.datasets import (
    PRODUCTION_REFERENCE,
    SOURCE_SHA256,
    ReferenceContractError,
    build_template_reference,
)

G4 = np.array([[0.0, 0.0], [0.0, 20.0], [0.0, 40.0], [0.0, 60.0]])
IDS4 = np.array([0, 1, 2, 3], np.int64)

TINY = dict(stop=20, blk=10, n_units=3, window_left=-2, window_right=3,
            max_snippets=4, local_channels=2, min_snippets=4)


def make_raw(length, spike_locals, ch=0, trough=-100.0, n=4):
    raw = np.zeros((length, n), np.float32)
    for s in spike_locals:
        raw[s, ch] = trough
    return raw


def tiny_factory(chunks):
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return iter(list(chunks))

    factory.calls = calls
    return factory


def run(chunks, t, u, **overrides):
    params = dict(TINY)
    params.update(overrides)
    return build_template_reference(tiny_factory(chunks), np.asarray(t, np.int64),
                                    np.asarray(u, np.int64), G4, IDS4, **params)


def test_uniform_rank_selection_and_valid_block():
    # spikes fully inside their chunk (window [t-2, t+3) within [0,10))
    chunks = [(0, make_raw(10, [2, 4, 5, 6])), (10, make_raw(10, [2, 4, 5, 6]))]
    t = [2, 4, 5, 6, 12, 14, 15, 16]
    u = [0] * 8
    arrays, diag = run(chunks, t, u)
    # 4 spikes per block, 2 per parity, all selected by the uniform targets.
    assert arrays["block_valid"][0, 0] and arrays["block_valid"][1, 0]
    assert arrays["block_channel"][0, 0] == 0 and arrays["block_channel"][1, 0] == 0
    assert np.all(arrays["estimated_local_channel"] == 0)
    assert diag["window_samples"] == [-2, 2]
    assert diag["minimum_snippets"] == 4
    assert diag["maximum_snippets_per_time_parity"] == 4
    assert diag["local_channels"] == 2


def test_single_snippet_unit_is_unknown():
    chunks = [(0, make_raw(10, [2, 3, 4, 5, 6], ch=0)), (10, make_raw(10, [2, 4, 5, 6]))]
    t = [2, 3, 4, 5, 6, 12, 14, 15, 16]
    u = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    arrays, diag = run(chunks, t, u)
    assert not arrays["block_valid"][:, 1].any()
    assert np.all(arrays["block_channel"][:, 1] == -1)
    assert arrays["estimated_local_channel"][1] == -1  # the unit-1 event (t=3)
    assert arrays["fallback_distance_blocks"][0, 1] == -1
    assert diag["unknown_event_fraction"] == pytest.approx(1 / 9)
    assert arrays["block_fallback"][0, 1]


def test_nearest_valid_fallback_with_lower_index_tie():
    params = dict(TINY); params.update(stop=30)
    chunks = [(0, make_raw(10, [2, 4, 5, 6])), (10, make_raw(10, [])),
              (20, make_raw(10, [2, 4, 5, 6]))]
    # unit 2 valid in blocks 0 and 2; block 1 falls back, tie -> block 0.
    t = [2, 4, 5, 6, 22, 24, 25, 26]
    u = [2] * 8
    arrays, _ = run(chunks, t, u, **params)
    assert arrays["block_valid"][0, 2] and arrays["block_valid"][2, 2]
    assert not arrays["block_valid"][1, 2]
    assert arrays["block_channel"][1, 2] == arrays["block_channel"][0, 2]
    assert arrays["fallback_distance_blocks"][1, 2] == 1
    assert arrays["block_fallback"][1, 2]


def test_boundary_window_dropped_but_rank_consumed_per_visit():
    # t=8 window [6,11) spans the chunk boundary [0,10): it is dropped but
    # visited by BOTH chunks, consuming one rank increment per visit (the
    # second visit flips its parity). The other four spikes form a valid
    # cell; nothing is excluded or "fixed".
    chunks = [(0, make_raw(10, [2, 4, 5, 6, 8])), (10, make_raw(10, []))]
    t = [2, 4, 5, 6, 8]
    u = [0] * 5
    arrays, diag = run(chunks, t, u)
    assert arrays["block_valid"][0, 0]  # stored t=2,5 (par0) and 4,6 (par1)
    assert arrays["estimated_local_channel"][-1] == 0  # t=8 still labeled from the block table
    # gaps 2,1,1,2 all < R-L=5 -> every spike flagged as same-unit overlap
    assert diag["same_unit_overlap_fraction"] == 1.0
    assert diag["no_global_collision_exclusion"] is True  # computed, never applied


def test_boundary_double_rank_changes_parity_selection():
    # t=9 sits on the boundary: visited by both chunks, stores nothing, and
    # consumes two rank increments in its block. t=15 is then occurrence
    # rank 0 of block 1 and is stored; both cells stay invalid (<4 snippets).
    chunks = [(0, make_raw(10, [9])), (10, make_raw(10, [5]))]
    t = [9, 15]
    u = [0, 0]
    arrays, _ = run(chunks, t, u)
    assert not arrays["block_valid"].any()
    assert np.all(arrays["estimated_local_channel"] == -1)


def test_odd_even_diagnostic_when_both_parities_filled():
    chunks = [(0, make_raw(10, [2, 4, 5, 6])), (10, make_raw(10, [2, 4, 5, 6]))]
    t = [2, 4, 5, 6, 12, 14, 15, 16]
    u = [0] * 8
    _, diag = run(chunks, t, u)
    assert diag["odd_even"]["eligible_cells"] == 2
    assert diag["odd_even"]["same_channel_fraction"] == 1.0
    assert diag["odd_even"]["median_spatial_delta_um"] == 0.0


def test_factory_is_called_once_per_pass_and_is_deterministic():
    chunks = [(0, make_raw(10, [2, 4, 6, 8])), (10, make_raw(10, [2, 4, 6, 8]))]
    t = [2, 4, 6, 8, 12, 14, 16, 18]
    factory = tiny_factory(chunks)
    first, _ = build_template_reference(factory, np.asarray(t), np.asarray([0] * 8),
                                        G4, IDS4, **TINY)
    assert factory.calls["n"] == 2  # exactly two passes, one fresh iterable each
    second, _ = build_template_reference(factory, np.asarray(t), np.asarray([0] * 8),
                                         G4, IDS4, **TINY)
    for key in first:
        assert np.array_equal(first[key], second[key])


def test_trough_channel_selection_follows_geometry():
    # Trough on channel 2 instead of 0: rough/local/median chain must pick it.
    chunks = [(0, make_raw(10, [2, 4, 5, 6], ch=2))]
    t = [2, 4, 5, 6]
    arrays, _ = run(chunks, t, [0] * 4, stop=10)
    assert arrays["block_channel"][0, 0] == 2
    assert np.all(arrays["estimated_local_channel"] == 2)


def test_builder_input_guards():
    chunks = [(0, make_raw(10, [2, 4, 5, 6]))]
    with pytest.raises(ReferenceContractError):
        run(chunks, [], [])  # empty GT rejected
    bad_columns = [(0, np.zeros((10, 5), np.float32))]
    with pytest.raises(ReferenceContractError):
        run(bad_columns, [2, 4, 5, 6], [0] * 4)
    with pytest.raises(ReferenceContractError):
        run(chunks, [4, 2], [0, 0])  # unsorted
    with pytest.raises(ReferenceContractError):
        run(chunks, [25], [0])  # beyond stop
    with pytest.raises(ReferenceContractError):
        run(chunks, [2], [3])  # unit id out of range


def test_production_constants_and_source_hash_are_locked():
    assert PRODUCTION_REFERENCE == {
        "stop": 9_000_000, "blk": 300_000, "n_units": 1200,
        "window_left": -50, "window_right": 71,
        "max_snippets": 16, "local_channels": 13, "min_snippets": 4,
    }
    assert SOURCE_SHA256 == "d312d459616ddba090cec6a098f9b47efe169086335b645a9db6a5ed7bffccee"


def test_label_exposure_statement_is_scoring_only():
    from shift_aso.neuropixels.datasets import LABEL_EXPOSURE
    assert "scorer-only" in LABEL_EXPOSURE
    assert "never CAL/DEV detector inputs" in LABEL_EXPOSURE
    assert "waveform-bank index" in LABEL_EXPOSURE
