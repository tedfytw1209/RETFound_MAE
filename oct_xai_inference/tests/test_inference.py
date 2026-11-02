import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import predict
from octxai.model import card_from_training_args, load_model
from octxai.pipeline import collect_items
from octxai.preprocess import build_transform


def run_cli(*argv):
    assert predict.main([str(a) for a in argv]) == 0


@pytest.fixture(params=["retfound_dir", "smp_dir"])
def model_dir(request):
    return request.getfixturevalue(request.param)


def test_transform_shapes():
    wide = Image.new("RGB", (1024, 496))
    assert tuple(build_transform(224)(wide).shape) == (3, 224, 224)
    assert tuple(build_transform(512)(wide).shape) == (3, 512, 512)


def test_cli_end_to_end(model_dir, images, tmp_path):
    out = tmp_path / "out" / "preds.csv"
    run_cli("-m", model_dir, "-i", images, "-o", out, "--device", "cpu", "--batch_size", 2)
    df = pd.read_csv(out)
    assert len(df) == 3
    assert list(df["status"]) == ["ok"] * 3
    p = df[["label_0_prob", "label_1_prob"]].to_numpy()
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert (df["pred_label"] == (df["label_1_prob"] >= 0.5).astype(int)).all()
    assert set(df["pred_class"]) <= {"Control", "Disease"}
    size = load_model(str(model_dir), "cpu").input_size
    for _, row in df.iterrows():
        cam = np.load(out.parent / row["saliency_npy"])
        assert cam.shape == (size, size) and cam.dtype == np.float32
        assert 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-6
        assert (out.parent / row["saliency_png"]).is_file()
        assert row["xai_class"] == row["pred_class"]


def test_saliency_does_not_change_probabilities(model_dir, images, tmp_path):
    run_cli("-m", model_dir, "-i", images, "-o", tmp_path / "a.csv", "--device", "cpu")
    run_cli("-m", model_dir, "-i", images, "-o", tmp_path / "b.csv", "--device", "cpu", "--no_saliency")
    a, b = pd.read_csv(tmp_path / "a.csv"), pd.read_csv(tmp_path / "b.csv")
    assert np.allclose(a["label_1_prob"], b["label_1_prob"], atol=1e-6)
    assert b["saliency_npy"].isna().all()
    assert not (tmp_path / "b_saliency").exists()


def test_batch_size_does_not_change_probabilities(model_dir, images, tmp_path):
    run_cli("-m", model_dir, "-i", images, "-o", tmp_path / "a.csv", "--device", "cpu", "--no_saliency",
            "--batch_size", 1)
    run_cli("-m", model_dir, "-i", images, "-o", tmp_path / "b.csv", "--device", "cpu", "--no_saliency",
            "--batch_size", 3)
    a, b = pd.read_csv(tmp_path / "a.csv"), pd.read_csv(tmp_path / "b.csv")
    assert np.allclose(a["label_1_prob"], b["label_1_prob"], atol=1e-5)


@pytest.mark.parametrize("method", ["hirescam", "gradcamv2", "gradcam++"])
def test_xai_methods_and_target_class(smp_dir, images, tmp_path, method):
    out = tmp_path / "p.csv"
    run_cli("-m", smp_dir, "-i", images / "scan_0.png", "-o", out, "--device", "cpu",
            "--xai", method, "--xai_target", 0)
    row = pd.read_csv(out).iloc[0]
    assert row["xai_class"] == "Control"
    assert np.isfinite(np.load(tmp_path / row["saliency_npy"])).all()


def test_bad_image_gets_a_row(retfound_dir, images, tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    out = tmp_path / "p.csv"
    run_cli("-m", retfound_dir, "-i", images / "scan_0.png", bad, "-o", out, "--device", "cpu")
    df = pd.read_csv(out)
    assert list(df["image_name"]) == ["scan_0.png", "broken.png"]
    assert df.loc[0, "status"] == "ok"
    assert df.loc[1, "status"].startswith("skipped")
    assert np.isnan(df.loc[1, "label_1_prob"])


def test_csv_input_and_duplicate_names(retfound_dir, images, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for d in "ab":
        Image.new("L", (80, 80)).save(tmp_path / d / "same.png")
    csv = tmp_path / "list.csv"
    pd.DataFrame({"image_path": ["a/same", "b/same.png"]}).to_csv(csv, index=False)
    items = collect_items([str(csv)], image_root=str(tmp_path))
    assert items[0].path.name == "same.jpg"  # extension-less names are .jpg, as in the training CSVs
    pd.DataFrame({"image_path": ["a/same.png", "b/same.png"]}).to_csv(csv, index=False)
    out = tmp_path / "p.csv"
    run_cli("-m", retfound_dir, "-i", csv, "--image_root", tmp_path, "-o", out, "--device", "cpu")
    df = pd.read_csv(out)
    assert list(df["status"]) == ["ok", "ok"]
    assert df["saliency_npy"].nunique() == 2  # same file name in two folders must not overwrite


def test_wrong_architecture_is_rejected(smp_dir, tmp_path):
    card = json.loads((smp_dir / "model_card.json").read_text())
    card["smp"]["fusion_dim"] = 16
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "model.pth").write_bytes((smp_dir / "model.pth").read_bytes())
    (bad / "model_card.json").write_text(json.dumps(card))
    with pytest.raises((ValueError, RuntimeError)):
        load_model(str(bad), "cpu")


def test_card_from_training_args_matches_the_kdd_smp_fuse_flags():
    args = dict(model="SMP", nb_classes=2, input_size=512, task="x", SMPMode="fuse", smp_fuse_mode="weighted_sum",
                smp_learnable_alpha=True, smp_alpha=0.5, smp_alpha_type="scalar", fusion_dim=9, align="pre",
                smp_size_match="decoder_to_encoder", seg_mask=True, mask_softmax=False, ignore_background=False,
                enc_idx=-2, dec_idx=-1, smp_classifier="conv")
    card = card_from_training_args(args, ["Control", "DME"], name="m")
    assert card["arch"] == "smp" and card["input_size"] == 512 and card["class_names"] == ["Control", "DME"]
    from octxai.model import SMP_DEFAULTS
    assert all(card["smp"][k] == v for k, v in SMP_DEFAULTS.items() if k not in
               ("seg_arch", "encoder", "in_channels", "seg_classes", "seg_activation", "dropout"))
    with pytest.raises(ValueError):
        card_from_training_args({**args, "add_mask": True})
    with pytest.raises(ValueError):
        card_from_training_args({**args, "model": "resnet-50"})


def test_load_raw_training_checkpoint(retfound_dir, tmp_path):
    from argparse import Namespace
    state = torch.load(retfound_dir / "model.pth", weights_only=True)["model"]
    ckpt = tmp_path / "checkpoint-best.pth"
    torch.save({"model": state, "epoch": 3, "args": Namespace(model="RETFound_mae", nb_classes=2, input_size=224,
                                                               task="t")}, ckpt)
    m = load_model(str(ckpt), "cpu", class_names=["N", "D"])
    assert m.arch == "retfound" and m.class_names == ["N", "D"]
    p = m.predict_proba(torch.zeros(1, 3, 224, 224))
    assert p.shape == (1, 2)
