"""Tests for filters/chain.py (Phase 4): CONTRACTS.md §12, §13.

Uses the `fake_backend`/`fs`/`native_backend` fixtures from conftest.py --
`fake_backend` for pure dispatch/plumbing checks (no compiled library
needed), `native_backend` (real, compiled) for numeric cascade/response
cross-checks, matching the convention in tests/test_error_analysis.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import freqz

from error_analysis import combined_response_error, response_error
from filters.chain import DEFAULT_PARAMS, ChainBlock, FilterChain

FS = 13333.0


# --- defaults and block creation -------------------------------------------


def test_default_params_valid_for_every_supported_fs(fs):
    chain = FilterChain(fs=fs)
    for kind in ("LP", "HP", "BP", "AP", "PK"):
        bid = chain.add_block(kind)
        block = chain.get_block(bid)
        assert block.is_valid, block.error
        assert block.kind == kind
        assert dict(block.params) == DEFAULT_PARAMS[kind]
        assert block.filter.kind == kind


def test_add_block_overrides_only_given_params():
    chain = FilterChain(fs=FS)
    bid = chain.add_block("BP", f_low=1500.0)
    block = chain.get_block(bid)
    assert dict(block.params) == {"f_low": 1500.0, "f_high": DEFAULT_PARAMS["BP"]["f_high"]}
    assert block.is_valid


def test_add_block_unknown_kind_raises():
    chain = FilterChain(fs=FS)
    with pytest.raises(ValueError):
        chain.add_block("XX")


def test_add_block_with_invalid_override_is_added_but_marked_invalid():
    chain = FilterChain(fs=FS)
    bid = chain.add_block("LP", fc=999_999.0)
    block = chain.get_block(bid)
    assert block in chain.blocks
    assert not block.is_valid
    assert block.error is not None
    assert dict(block.params) == {"fc": 999_999.0}


def test_block_params_are_read_only():
    chain = FilterChain(fs=FS)
    bid = chain.add_block("LP", fc=1000.0)
    block = chain.get_block(bid)
    with pytest.raises(TypeError):
        block.params["fc"] = 5000.0


# --- stable IDs after reorder/delete ----------------------------------------


def test_ids_stable_after_reorder_and_delete():
    chain = FilterChain(fs=FS)
    id_a = chain.add_block("LP", fc=1000.0)
    id_b = chain.add_block("HP", fc=1000.0)
    id_c = chain.add_block("AP", fc=1000.0, Q=1.0)

    assert [b.id for b in chain.blocks] == [id_a, id_b, id_c]

    chain.move_block(id_c, 0)
    assert [b.id for b in chain.blocks] == [id_c, id_a, id_b]
    assert chain.get_block(id_a).kind == "LP"

    chain.remove_block(id_a)
    assert [b.id for b in chain.blocks] == [id_c, id_b]
    assert chain.get_block(id_c).kind == "AP"
    assert chain.get_block(id_b).kind == "HP"

    new_id = chain.add_block("LP", fc=1000.0)
    assert new_id not in (id_a, id_b, id_c)


# --- add/remove/reorder/clear -----------------------------------------------


def test_add_remove_reorder_clear():
    chain = FilterChain(fs=FS)
    id1 = chain.add_block("LP")
    id2 = chain.add_block("HP")
    assert len(chain.blocks) == 2

    chain.move_block(id2, 0)
    assert [b.id for b in chain.blocks] == [id2, id1]

    chain.remove_block(id1)
    assert [b.id for b in chain.blocks] == [id2]

    chain.clear()
    assert chain.blocks == []
    assert chain.fs == FS  # CONTRACTS.md §13: Clear leaves fs unchanged


def test_move_block_clamps_out_of_range_index():
    chain = FilterChain(fs=FS)
    id1 = chain.add_block("LP")
    id2 = chain.add_block("HP")
    chain.move_block(id1, 999)
    assert [b.id for b in chain.blocks] == [id2, id1]
    chain.move_block(id1, -5)
    assert [b.id for b in chain.blocks] == [id1, id2]


def test_remove_move_update_unknown_id_raises():
    chain = FilterChain(fs=FS)
    with pytest.raises(KeyError):
        chain.remove_block("nope")
    with pytest.raises(KeyError):
        chain.move_block("nope", 0)
    with pytest.raises(KeyError):
        chain.update_params("nope", fc=1000.0)
    with pytest.raises(KeyError):
        chain.get_block("nope")


# --- selection fallback ------------------------------------------------------


def test_selection_defaults_to_combined():
    chain = FilterChain(fs=FS)
    assert chain.selected_id is None
    assert chain.selected_block is None


def test_select_and_selection_fallback_on_delete():
    chain = FilterChain(fs=FS)
    id1 = chain.add_block("LP")
    id2 = chain.add_block("HP")
    chain.select(id1)
    assert chain.selected_id == id1
    assert chain.selected_block.id == id1

    chain.remove_block(id1)
    assert chain.selected_id is None
    assert chain.selected_block is None

    chain.select(id2)
    assert chain.selected_id == id2


def test_select_unknown_id_raises():
    chain = FilterChain(fs=FS)
    with pytest.raises(KeyError):
        chain.select("nope")


def test_select_combined_explicitly():
    chain = FilterChain(fs=FS)
    id1 = chain.add_block("LP")
    chain.select(id1)
    chain.select(None)
    assert chain.selected_id is None


def test_clear_falls_back_selection_to_combined():
    chain = FilterChain(fs=FS)
    id1 = chain.add_block("LP")
    chain.select(id1)
    chain.clear()
    assert chain.selected_id is None


# --- dirty-state transitions --------------------------------------------------


def test_dirty_state_transitions():
    chain = FilterChain(fs=FS)
    assert chain.dirty is False

    id1 = chain.add_block("LP")
    assert chain.dirty is True
    chain.mark_clean()
    assert chain.dirty is False

    chain.update_params(id1, fc=2000.0)
    assert chain.dirty is True
    chain.mark_clean()

    chain.move_block(id1, 0)
    assert chain.dirty is True
    chain.mark_clean()

    id2 = chain.add_block("HP")
    chain.mark_clean()
    chain.remove_block(id2)
    assert chain.dirty is True
    chain.mark_clean()

    chain.fs = 20_000.0
    assert chain.dirty is True
    chain.mark_clean()

    chain.clear()
    assert chain.dirty is True


def test_selection_does_not_mark_dirty():
    chain = FilterChain(fs=FS)
    id1 = chain.add_block("LP")
    chain.mark_clean()
    chain.select(id1)
    assert chain.dirty is False
    chain.select(None)
    assert chain.dirty is False


# --- fs changes and invalidation ---------------------------------------------


def test_fs_change_invalidates_and_preserves_params():
    chain = FilterChain(fs=40_000.0)
    bid = chain.add_block("LP", fc=15_000.0)  # valid: fc_max(40_000) = 18_000
    assert chain.get_block(bid).is_valid

    chain.fs = 5_000.0  # fc_max(5_000) = 2_250 -> now invalid
    block = chain.get_block(bid)
    assert not block.is_valid
    assert block.error is not None
    assert dict(block.params) == {"fc": 15_000.0}  # not clamped

    chain.fs = 40_000.0  # revert -> valid again, params untouched throughout
    block = chain.get_block(bid)
    assert block.is_valid
    assert block.filter.fc == pytest.approx(15_000.0)


def test_fs_setter_rejects_out_of_range():
    chain = FilterChain(fs=FS)
    with pytest.raises(ValueError):
        chain.fs = 1_000.0
    with pytest.raises(ValueError):
        chain.fs = 50_000.0
    assert chain.fs == FS  # rejected write leaves fs unchanged


def test_constructor_rejects_out_of_range_fs():
    with pytest.raises(ValueError):
        FilterChain(fs=1_000.0)


def test_has_invalid_blocks_and_validation_messages():
    chain = FilterChain(fs=40_000.0)
    bid = chain.add_block("LP", fc=15_000.0)
    assert not chain.has_invalid_blocks
    assert chain.validation_messages() == {}

    chain.fs = 5_000.0
    assert chain.has_invalid_blocks
    messages = chain.validation_messages()
    assert set(messages) == {bid}
    assert "fc" in messages[bid]


# --- parameter updates ---------------------------------------------------------


def test_update_params_changes_filter():
    chain = FilterChain(fs=FS)
    bid = chain.add_block("LP", fc=1000.0)
    old_coeffs = chain.get_block(bid).filter.ideal_coefficients()

    chain.update_params(bid, fc=2000.0)
    block = chain.get_block(bid)
    assert block.is_valid
    assert dict(block.params) == {"fc": 2000.0}
    assert block.filter.ideal_coefficients() != old_coeffs


def test_update_params_invalid_marks_block_without_raising():
    chain = FilterChain(fs=FS)
    bid = chain.add_block("LP", fc=1000.0)

    chain.update_params(bid, fc=999_999.0)
    block = chain.get_block(bid)
    assert not block.is_valid
    assert block.error is not None
    assert dict(block.params) == {"fc": 999_999.0}


def test_update_params_partial_bp_preserves_other_field():
    chain = FilterChain(fs=FS)
    bid = chain.add_block("BP", f_low=1000.0, f_high=2000.0)
    chain.update_params(bid, f_low=1500.0)
    block = chain.get_block(bid)
    assert dict(block.params) == {"f_low": 1500.0, "f_high": 2000.0}
    assert block.is_valid


# --- empty-chain handling -------------------------------------------------------


def test_empty_chain_has_no_blocks_and_no_valid_filters():
    chain = FilterChain(fs=FS)
    assert chain.blocks == []
    assert chain.valid_filters == []
    assert not chain.has_invalid_blocks


def test_combined_response_empty_chain_raises():
    chain = FilterChain(fs=FS)
    with pytest.raises(ValueError, match="empty"):
        chain.combined_ideal_response(np.array([1000.0]))


def test_combined_response_all_invalid_blocks_raises():
    chain = FilterChain(fs=40_000.0)
    chain.add_block("LP", fc=15_000.0)  # valid at fs=40_000
    chain.fs = 5_000.0  # now invalid
    assert chain.has_invalid_blocks
    with pytest.raises(ValueError, match="invalid"):
        chain.combined_ideal_response(np.array([500.0]))


def test_combined_q14_response_missing_backend_raises():
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=3000.0)
    with pytest.raises(ValueError, match="NativeBackend"):
        chain.combined_q14_response(np.array([1000.0]), None)


# --- cascade equivalence for one block ------------------------------------------


def test_combined_response_single_block_matches_direct_response():
    freq = np.linspace(200.0, 6000.0, 30)
    chain = FilterChain(fs=FS)
    bid = chain.add_block("LP", fc=3000.0)
    result = chain.combined_ideal_response(freq)
    direct = chain.get_block(bid).filter.ideal_response(freq)
    assert result.magnitude_db == pytest.approx(direct.magnitude_db)
    assert result.phase_deg == pytest.approx(direct.phase_deg)


# --- multi-block dB and phase composition ---------------------------------------


def test_combined_ideal_response_is_db_additive_and_unwrapped_after_sum():
    freq = np.linspace(200.0, 6000.0, 40)
    chain = FilterChain(fs=FS)
    id_lp = chain.add_block("LP", fc=4000.0)
    id_hp = chain.add_block("HP", fc=1000.0)
    result = chain.combined_ideal_response(freq)

    resp_lp = chain.get_block(id_lp).filter.ideal_response(freq)
    resp_hp = chain.get_block(id_hp).filter.ideal_response(freq)

    expected_mag = resp_lp.magnitude_db + resp_hp.magnitude_db
    expected_phase = np.degrees(np.unwrap(np.radians(resp_lp.phase_deg + resp_hp.phase_deg)))

    assert result.magnitude_db == pytest.approx(expected_mag)
    assert result.phase_deg == pytest.approx(expected_phase)


def test_combined_magnitude_equals_product_of_transfer_functions():
    # Independent cross-check of CONTRACTS.md §12's H_combined = Pi H_i,
    # verifying the dB-additive shortcut against literal complex-transfer-
    # function multiplication (not just re-deriving the same formula).
    freq = np.linspace(200.0, 6000.0, 40)
    chain = FilterChain(fs=FS)
    id_lp = chain.add_block("LP", fc=4000.0)
    id_hp = chain.add_block("HP", fc=1000.0)
    result = chain.combined_ideal_response(freq)

    lp_c = chain.get_block(id_lp).filter.ideal_coefficients()
    hp_c = chain.get_block(id_hp).filter.ideal_coefficients()
    w = 2.0 * np.pi * freq / FS
    _, h_lp = freqz(*lp_c.as_ba(), worN=w)
    _, h_hp = freqz(*hp_c.as_ba(), worN=w)
    expected_mag_db = 20.0 * np.log10(np.abs(h_lp * h_hp))

    assert result.magnitude_db == pytest.approx(expected_mag_db, abs=1e-9)


def test_combined_q14_response_matches_manual_sum(native_backend):
    freq = np.linspace(200.0, 6000.0, 20)
    chain = FilterChain(fs=FS)
    id_lp = chain.add_block("LP", fc=4000.0)
    id_hp = chain.add_block("HP", fc=1000.0)
    result = chain.combined_q14_response(freq, native_backend)

    resp_lp = chain.get_block(id_lp).filter.q14_response(freq, native_backend)
    resp_hp = chain.get_block(id_hp).filter.q14_response(freq, native_backend)
    expected_mag = resp_lp.magnitude_db + resp_hp.magnitude_db

    assert result.magnitude_db == pytest.approx(expected_mag)


# --- compatibility with response_error() / combined_response_error() -----------


def test_valid_filters_compatible_with_combined_response_error(native_backend):
    chain = FilterChain(fs=FS)
    chain.add_block("LP", fc=4000.0)
    chain.add_block("HP", fc=1000.0)

    result = combined_response_error(chain.valid_filters, native_backend)
    assert result.max_db < 5.0
    assert result.rms_db <= result.max_db


def test_valid_filters_excludes_invalid_blocks_for_combined_response_error(native_backend):
    # fc_max(10_000) = 4_500: a 4_000 Hz block stays valid, a 17_999 Hz block
    # (valid up front at fs=40_000, fc_max=18_000) does not.
    chain = FilterChain(fs=40_000.0)
    chain.add_block("LP", fc=4000.0)
    bad_id = chain.add_block("LP", fc=17_999.0)
    chain.fs = 10_000.0
    assert not chain.get_block(bad_id).is_valid

    result = combined_response_error(chain.valid_filters, native_backend)
    assert len(chain.valid_filters) == 1
    assert result.max_db < 5.0


def test_block_filter_compatible_with_response_error(native_backend):
    chain = FilterChain(fs=FS)
    bid = chain.add_block("AP", fc=2000.0, Q=1.0)
    block = chain.get_block(bid)
    result = response_error(block.filter, native_backend)
    assert result.max_db >= 0.0


# --- enabled/disabled (bypass) state --------------------------------------------


def test_new_block_defaults_enabled():
    chain = FilterChain(fs=FS)
    bid = chain.add_block("LP")
    assert chain.get_block(bid).enabled is True


def test_set_enabled_toggles_state_and_marks_dirty():
    chain = FilterChain(fs=FS)
    bid = chain.add_block("LP")
    chain.mark_clean()

    chain.set_enabled(bid, False)
    assert chain.get_block(bid).enabled is False
    assert chain.dirty is True

    chain.mark_clean()
    chain.set_enabled(bid, True)
    assert chain.get_block(bid).enabled is True
    assert chain.dirty is True


def test_set_enabled_noop_does_not_mark_dirty():
    chain = FilterChain(fs=FS)
    bid = chain.add_block("LP")
    chain.mark_clean()

    chain.set_enabled(bid, True)  # already enabled -- no-op

    assert chain.dirty is False


def test_set_enabled_unknown_id_raises():
    chain = FilterChain(fs=FS)
    with pytest.raises(KeyError):
        chain.set_enabled("nope", False)


def test_set_enabled_does_not_touch_params_or_validity():
    chain = FilterChain(fs=FS)
    bid = chain.add_block("LP", fc=999_999.0)  # invalid on purpose
    chain.set_enabled(bid, False)
    block = chain.get_block(bid)
    assert not block.is_valid
    assert block.error is not None
    assert dict(block.params) == {"fc": 999_999.0}


def test_disabled_block_survives_fs_change_and_update_params():
    chain = FilterChain(fs=FS)
    bid = chain.add_block("LP", fc=1000.0)
    chain.set_enabled(bid, False)

    chain.fs = 20_000.0
    assert chain.get_block(bid).enabled is False

    chain.update_params(bid, fc=2000.0)
    assert chain.get_block(bid).enabled is False
    assert dict(chain.get_block(bid).params) == {"fc": 2000.0}


def test_disabled_block_excluded_from_valid_filters():
    chain = FilterChain(fs=FS)
    id1 = chain.add_block("LP", fc=1000.0)
    id2 = chain.add_block("HP", fc=1000.0)
    chain.set_enabled(id2, False)

    assert len(chain.valid_filters) == 1
    assert chain.valid_filters[0] is chain.get_block(id1).filter


def test_disabled_invalid_block_does_not_set_has_invalid_blocks():
    chain = FilterChain(fs=FS)
    bid = chain.add_block("LP", fc=999_999.0)  # invalid
    assert chain.has_invalid_blocks

    chain.set_enabled(bid, False)
    assert not chain.has_invalid_blocks  # bypassed -- no longer blocks export/toolbar


def test_enabled_invalid_block_still_sets_has_invalid_blocks():
    chain = FilterChain(fs=FS)
    valid_id = chain.add_block("LP", fc=1000.0)
    bad_id = chain.add_block("HP", fc=999_999.0)
    chain.set_enabled(valid_id, False)

    assert chain.has_invalid_blocks  # bad_id is still enabled and invalid


def test_combined_response_all_disabled_blocks_raises_like_all_invalid():
    chain = FilterChain(fs=FS)
    bid = chain.add_block("LP", fc=1000.0)
    chain.set_enabled(bid, False)

    with pytest.raises(ValueError, match="invalid or disabled"):
        chain.combined_ideal_response(np.array([500.0]))


def test_combined_response_excludes_disabled_block(native_backend):
    freq = np.linspace(200.0, 6000.0, 20)
    chain = FilterChain(fs=FS)
    id_lp = chain.add_block("LP", fc=4000.0)
    id_hp = chain.add_block("HP", fc=1000.0)
    chain.set_enabled(id_hp, False)

    result = chain.combined_ideal_response(freq)
    direct = chain.get_block(id_lp).filter.ideal_response(freq)

    assert result.magnitude_db == pytest.approx(direct.magnitude_db)
