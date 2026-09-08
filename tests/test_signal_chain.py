"""Tests for signals/chain.py: the additive Time-Domain source-generator model (docs/CONCEPT.md §11.2)."""

from __future__ import annotations

import numpy as np
import pytest

from signals import DEFAULT_SIGNAL_PARAMS, SignalChain

FS = 13_333.0


# --- defaults and block creation -------------------------------------------


def test_default_params_valid_for_every_kind():
    chain = SignalChain()
    for kind in ("SIN", "DC", "NOISE"):
        bid = chain.add_block(kind)
        block = chain.get_block(bid)
        assert block.is_valid, block.error
        assert block.kind == kind
        assert block.factor == 1.0
        for key, value in DEFAULT_SIGNAL_PARAMS[kind].items():
            assert block.params[key] == value


def test_add_block_unknown_kind_raises():
    chain = SignalChain()
    with pytest.raises(ValueError):
        chain.add_block("XX")


def test_add_block_overrides_only_given_params():
    chain = SignalChain()
    bid = chain.add_block("SIN", frequency=2_000.0)
    block = chain.get_block(bid)
    assert block.params["frequency"] == 2_000.0
    assert block.params["amplitude"] == DEFAULT_SIGNAL_PARAMS["SIN"]["amplitude"]


def test_add_block_custom_factor():
    chain = SignalChain()
    bid = chain.add_block("DC", factor=0.5, value=1.0)
    block = chain.get_block(bid)
    assert block.factor == 0.5


def test_add_block_with_invalid_params_is_added_but_marked_invalid():
    chain = SignalChain()
    bid = chain.add_block("SIN", frequency=-1.0)
    block = chain.get_block(bid)
    assert block in chain.blocks
    assert not block.is_valid
    assert block.error is not None
    assert block.params["frequency"] == -1.0


def test_noise_negative_amplitude_is_invalid():
    chain = SignalChain()
    bid = chain.add_block("NOISE", amplitude=-0.1)
    assert not chain.get_block(bid).is_valid


def test_block_params_are_read_only():
    chain = SignalChain()
    bid = chain.add_block("SIN")
    with pytest.raises(TypeError):
        chain.get_block(bid).params["frequency"] = 5.0


def test_get_block_unknown_id_raises_keyerror():
    chain = SignalChain()
    with pytest.raises(KeyError):
        chain.get_block("nope")


# --- mutation ----------------------------------------------------------------


def test_remove_block():
    chain = SignalChain()
    bid = chain.add_block("DC")
    chain.remove_block(bid)
    assert chain.blocks == []
    with pytest.raises(KeyError):
        chain.get_block(bid)


def test_update_params_merges_and_revalidates():
    chain = SignalChain()
    bid = chain.add_block("SIN", frequency=1_000.0)
    chain.update_params(bid, frequency=500.0)
    block = chain.get_block(bid)
    assert block.params["frequency"] == 500.0
    assert block.params["amplitude"] == DEFAULT_SIGNAL_PARAMS["SIN"]["amplitude"]
    assert block.is_valid


def test_update_params_can_make_a_valid_block_invalid_without_raising():
    chain = SignalChain()
    bid = chain.add_block("SIN")
    chain.update_params(bid, frequency=-5.0)
    assert not chain.get_block(bid).is_valid


def test_set_factor_does_not_touch_params_or_design_identity():
    chain = SignalChain()
    bid = chain.add_block("DC", value=1.0)
    design_before = chain.get_block(bid).design
    chain.set_factor(bid, 2.0)
    block = chain.get_block(bid)
    assert block.factor == 2.0
    assert block.design is design_before


def test_clear():
    chain = SignalChain()
    chain.add_block("DC")
    chain.add_block("SIN")
    chain.clear()
    assert chain.blocks == []


# --- validity introspection --------------------------------------------------


def test_has_invalid_blocks_and_validation_messages():
    chain = SignalChain()
    good = chain.add_block("DC", value=0.5)
    bad = chain.add_block("SIN", frequency=0.0)
    assert chain.has_invalid_blocks
    messages = chain.validation_messages()
    assert set(messages) == {bad}
    assert good not in messages


# --- noise reproducibility ----------------------------------------------------


def test_noise_seed_is_stable_across_generate_calls():
    chain = SignalChain()
    bid = chain.add_block("NOISE", amplitude=0.2)
    block = chain.get_block(bid)
    first = block.design.generate(100, FS)
    second = block.design.generate(100, FS)
    np.testing.assert_array_equal(first, second)


def test_noise_seed_is_preserved_across_unrelated_param_edits():
    chain = SignalChain()
    bid = chain.add_block("NOISE", amplitude=0.2)
    seed_before = chain.get_block(bid).design.seed
    chain.set_factor(bid, 0.5)
    assert chain.get_block(bid).design.seed == seed_before


def test_two_noise_blocks_get_different_seeds():
    chain = SignalChain()
    b1 = chain.add_block("NOISE")
    b2 = chain.add_block("NOISE")
    assert chain.get_block(b1).design.seed != chain.get_block(b2).design.seed


def test_explicit_seed_is_reproducible():
    chain = SignalChain()
    bid = chain.add_block("NOISE", amplitude=0.1, seed=42)
    a = chain.get_block(bid).design.generate(50, FS)
    chain2 = SignalChain()
    bid2 = chain2.add_block("NOISE", amplitude=0.1, seed=42)
    b = chain2.get_block(bid2).design.generate(50, FS)
    np.testing.assert_array_equal(a, b)


# --- reseed (explicit user action, docs/CONCEPT.md §11) -----------------------


def test_reseed_changes_the_seed():
    chain = SignalChain()
    bid = chain.add_block("NOISE", amplitude=0.2, seed=1)
    chain.reseed(bid)
    assert chain.get_block(bid).design.seed != 1


def test_reseed_preserves_other_params_and_stays_valid():
    chain = SignalChain()
    bid = chain.add_block("NOISE", amplitude=0.2)
    chain.reseed(bid)
    block = chain.get_block(bid)
    assert block.is_valid, block.error
    assert block.params["amplitude"] == 0.2


def test_reseed_marks_dirty():
    chain = SignalChain()
    bid = chain.add_block("NOISE")
    chain.mark_clean()
    chain.reseed(bid)
    assert chain.dirty is True


def test_reseed_on_non_noise_block_raises():
    chain = SignalChain()
    bid = chain.add_block("DC")
    with pytest.raises(ValueError, match="NOISE"):
        chain.reseed(bid)


def test_reseed_unknown_block_raises_keyerror():
    chain = SignalChain()
    with pytest.raises(KeyError):
        chain.reseed("nope")


def test_reseed_changes_generated_output():
    chain = SignalChain()
    bid = chain.add_block("NOISE", amplitude=0.2, seed=1)
    before = chain.get_block(bid).design.generate(50, FS)
    chain.reseed(bid)
    after = chain.get_block(bid).design.generate(50, FS)
    assert not np.array_equal(before, after)


# --- CSV import ----------------------------------------------------------------


def test_csv_import_resamples_onto_fs_grid(tmp_path):
    path = tmp_path / "signal.csv"
    path.write_text("0.0,0.0\n1.0,1.0\n2.0,0.0\n")
    chain = SignalChain()
    bid = chain.add_block("CSV", file_path=str(path))
    block = chain.get_block(bid)
    assert block.is_valid, block.error
    values = block.design.generate(3, fs=1.0)  # t = 0, 1, 2 seconds
    np.testing.assert_allclose(values, [0.0, 1.0, 0.0])


def test_csv_import_tolerates_header_row(tmp_path):
    path = tmp_path / "signal.csv"
    path.write_text("time,value\n0.0,0.0\n1.0,2.0\n")
    chain = SignalChain()
    bid = chain.add_block("CSV", file_path=str(path))
    assert chain.get_block(bid).is_valid


def test_csv_import_pads_outside_recorded_range_with_zero(tmp_path):
    path = tmp_path / "signal.csv"
    path.write_text("0.0,5.0\n1.0,5.0\n")
    chain = SignalChain()
    bid = chain.add_block("CSV", file_path=str(path))
    block = chain.get_block(bid)
    values = block.design.generate(4, fs=1.0)  # t = 0, 1, 2, 3
    np.testing.assert_allclose(values, [5.0, 5.0, 0.0, 0.0])


def test_csv_import_missing_file_is_invalid():
    chain = SignalChain()
    bid = chain.add_block("CSV", file_path="/no/such/file.csv")
    assert not chain.get_block(bid).is_valid


def test_csv_import_no_file_selected_is_invalid():
    chain = SignalChain()
    bid = chain.add_block("CSV")
    assert not chain.get_block(bid).is_valid


def test_csv_import_duplicate_timestamps_are_invalid(tmp_path):
    path = tmp_path / "signal.csv"
    path.write_text("0.0,0.0\n0.0,1.0\n1.0,2.0\n")
    chain = SignalChain()
    bid = chain.add_block("CSV", file_path=str(path))
    assert not chain.get_block(bid).is_valid


# --- source signal (summation) ------------------------------------------------


def test_source_signal_empty_chain_raises():
    chain = SignalChain()
    with pytest.raises(ValueError):
        chain.source_signal(10, FS)


def test_source_signal_all_invalid_raises():
    chain = SignalChain()
    chain.add_block("SIN", frequency=-1.0)
    with pytest.raises(ValueError):
        chain.source_signal(10, FS)


def test_source_signal_sums_valid_blocks_and_excludes_invalid():
    chain = SignalChain()
    chain.add_block("DC", value=1.0)
    chain.add_block("DC", value=2.0)
    chain.add_block("SIN", frequency=-1.0)  # invalid, excluded
    signal = chain.source_signal(5, FS)
    np.testing.assert_allclose(signal, np.full(5, 3.0))


def test_source_signal_applies_factor():
    chain = SignalChain()
    chain.add_block("DC", value=1.0, factor=0.25)
    signal = chain.source_signal(4, FS)
    np.testing.assert_allclose(signal, np.full(4, 0.25))


def test_source_signal_sine_matches_closed_form():
    chain = SignalChain()
    bid = chain.add_block("SIN", frequency=1_000.0, amplitude=0.5, phase_deg=0.0)
    n = 8
    signal = chain.source_signal(n, FS)
    t = np.arange(n) / FS
    expected = 0.5 * np.sin(2.0 * np.pi * 1_000.0 * t)
    np.testing.assert_allclose(signal, expected)


# --- dirty state (mirrors filters.chain.FilterChain, docs/CONTRACTS.md §15) --------


def test_starts_clean():
    assert SignalChain().dirty is False


def test_add_block_marks_dirty():
    chain = SignalChain()
    chain.add_block("DC")
    assert chain.dirty is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c, bid: c.remove_block(bid),
        lambda c, bid: c.update_params(bid, frequency=500.0),
        lambda c, bid: c.set_factor(bid, 0.5),
        lambda c, bid: c.clear(),
    ],
)
def test_every_other_mutation_marks_dirty(mutate):
    """Each method sets `dirty` on its own -- verified by clearing it right
    after setup (via a prior `add_block`), so a passing add_block alone
    can't hide a mutator that forgot to set the flag."""
    chain = SignalChain()
    bid = chain.add_block("SIN")
    chain.mark_clean()
    mutate(chain, bid)
    assert chain.dirty is True


def test_mark_clean_resets_dirty():
    chain = SignalChain()
    chain.add_block("DC")
    assert chain.dirty is True
    chain.mark_clean()
    assert chain.dirty is False
