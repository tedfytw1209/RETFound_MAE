# tests/test_multiseed_util.py
from util.multiseed import apply_seed_suffix, base_name, seed_suffix


def test_seed_suffix_empty_for_original_run():
    assert seed_suffix(split_seed=42, seed=0, default_seed=0) == ''
    assert seed_suffix(split_seed=42, seed=42, default_seed=42) == ''


def test_seed_suffix_split_and_train_modes():
    assert seed_suffix(split_seed=1952, seed=0, default_seed=0) == '-s1952'
    assert seed_suffix(split_seed=42, seed=1952, default_seed=0) == '-seed1952'


def test_apply_seed_suffix_keeps_trailing_slash():
    assert apply_seed_suffix('A-trsub0--/', 1952, 0, 0) == 'A-trsub0---s1952/'
    assert apply_seed_suffix('A-trsub0--/', 42, 0, 0) == 'A-trsub0--/'
    assert apply_seed_suffix('X-seed0', 1952, 0, 0) == 'X-seed0-s1952'


def test_base_name_strips_seed_tokens():
    assert base_name('A-trsub0--/') == 'A-trsub0--'
    assert base_name('A-trsub0---s1952/') == 'A-trsub0--'
    assert base_name('A-trsub0---seed1952/') == 'A-trsub0--'
    assert base_name('DME-v5-m-hirescam-224-blk-1-seed0-segmask-s1952') == 'DME-v5-m-hirescam-224-blk-1-segmask'
    assert base_name('DME-v5-m-hirescam-224-blk-1-seed1952-segmask') == 'DME-v5-m-hirescam-224-blk-1-segmask'
