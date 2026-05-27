"""Phase-2 FINAL — learned sequence-locator ⊕ kernel-GBDT. COLAB (GPU + high-RAM).

The last swing (per the agreed "succeed or harvest"). A SUPERVISED sequence aligner (BiGRU over post-PS
GR → locality cross-attention to the typewell ∪ prefix memory → drift, src/locator_model.SeqLocator),
trained end-to-end on TVT. Unlike the kernel's *unsupervised* DTW it optimises the real target — the bet
is a DECORRELATED member that earns blend weight. CPU smoke test was under-powered (≈init); this is the
real train (623 wells, GPU, full epochs).

Pipeline: train by-well CV → OOF/sacred/test → **OOF-calibrate** the locator (affine fit on dev-OOF →
sacred/test; robust to the sign/scale issue the smoke test showed) → blend with the kernel cat on dev-OOF
→ verdict on SACRED: does loc⊕kernel beat kernel-only (~9.16)? Env: ROGII_LOC_W, ROGII_LOC_EPOCHS.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import blend, cv, dashboard as dash, data, locator_data as ld, locator_model as lm, train
from src.evaluate import rmse
from src.observer import Experiment

CACHE = Path(os.environ.get("DRIVE_ROOT") or "data") / "cache"
VER = "v14_locator"
W = int(os.environ.get("ROGII_LOC_W") or 32)
EP = int(os.environ.get("ROGII_LOC_EPOCHS") or 25)
CAT = dict(depth=6, learning_rate=0.03, random_seed=42, l2_leaf_reg=2.0, min_data_in_leaf=15)


def main() -> None:
    t0 = time.time()
    dev_w, sac_w = cv.sacred_split(data.list_well_ids("train"))
    test_w = data.list_well_ids("test")
    dash.goal_banner(VER, f"learned seq-locator (W={W}, {EP}ep) ⊕ kernel",
                     "supervised aligner — does it ADD to 9.16 on sacred? (else harvest)")
    exp = Experiment.start(
        version=VER, parent="v6s8_fast",
        hypothesis=("A SUPERVISED sequence aligner (BiGRU + locality cross-attention to typewell∪prefix, "
                    "trained on TVT) is decorrelated from the kernel's unsupervised DTW; OOF-calibrated + "
                    "blended it lowers sacred below ~9.16. Final swing — else harvest."),
        predicted_delta=0.10, confidence="low",
        pipeline_changes=["learned sequence locator ⊕ kernel-GBDT"], cloud_or_local="cloud")

    # ---------- locator ----------
    dev_e = ld.build(dev_w, "train", W); sac_e = ld.build(sac_w, "train", W); test_e = ld.build(test_w, "test", W)
    print(f"[loc] episodes: dev {len(dev_e)} | sac {len(sac_e)} | test {len(test_e)}")
    loc_oof, loc_sac, loc_test, fr = lm.train_cv(dev_e, sac_e, test_e, n_folds=5, epochs=EP, d=64,
                                                 accum=2, mem_sub=4000, model_cls=lm.SeqLocator)

    # ---------- kernel GBDT (cached 222) ----------
    dev_df = pd.read_parquet(CACHE / "dev_k9.parquet"); sac_df = pd.read_parquet(CACHE / "sacred_k9.parquet")
    feats = [c for c in dev_df.columns if c not in {"well", "id", "target"}]
    yd = dev_df["target"].to_numpy(np.float32); gd = dev_df["well"].to_numpy(); ys = sac_df["target"].to_numpy(np.float32)
    r = train.train_variant(f"{VER}_cat", "cat", dev_df[feats].astype("float32"), yd, gd,
                            X_test=sac_df[feats].astype("float32"), params=CAT, save=False, use_gpu="auto")
    k_oof = dict(zip(dev_df["id"], r.oof)); k_sac = dict(zip(sac_df["id"], r.test_pred))

    # ---------- align by id ----------
    dev_ids = [i for i in dev_df["id"].tolist() if i in loc_oof]
    sac_ids = [i for i in sac_df["id"].tolist() if i in loc_sac]
    print(f"[align] dev {len(dev_ids)}/{len(dev_df)} | sac {len(sac_ids)}/{len(sac_df)}")
    y_dev = dev_df.set_index("id").loc[dev_ids, "target"].to_numpy(np.float32)
    y_sac = sac_df.set_index("id").loc[sac_ids, "target"].to_numpy(np.float32)
    pick = lambda by, ids: np.array([by[i] for i in ids], float)
    lo, ls = pick(loc_oof, dev_ids), pick(loc_sac, sac_ids)
    ko, ks = pick(k_oof, dev_ids), pick(k_sac, sac_ids)

    # ---------- OOF-calibrate the locator (affine; robust to sign/scale) ----------
    a, b = np.polyfit(lo, y_dev, 1)
    lo_c, ls_c = a * lo + b, a * ls + b
    loc_alone, loc_raw = rmse(y_sac, ls_c), rmse(y_sac, ls)
    print(f"[loc] calib a={a:.3f} b={b:.2f} | locator-alone sacred {loc_alone:.3f} (raw {loc_raw:.3f}) "
          f"| oof corr {np.corrcoef(lo, y_dev)[0,1]:.3f} | fold val {[round(x,2) for x in fr]}")

    # ---------- blend vs kernel-only ----------
    k_only = rmse(y_sac, ks)
    w, _, _ = blend.nm_optimize_oof({"loc": lo_c, "kernel": ko}, y_dev)
    ens = rmse(y_sac, blend.apply_blend({"loc": ls_c, "kernel": ks}, w))
    corr = float(np.corrcoef(y_dev - lo_c, y_dev - ko)[0, 1])
    print(f"\nkernel-only {k_only:.3f} | loc⊕kernel {ens:.3f} | loc weight {w.get('loc',0):.3f} | resid corr {corr:.3f}")

    out = CACHE / "loc_p2"; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "preds.npz", dev_ids=np.array(dev_ids), sac_ids=np.array(sac_ids),
             loc_oof_c=lo_c, loc_sac_c=ls_c, a=a, b=b)
    exp.record(oof_score_mean=ens, oof_score_per_fold=[float(x) for x in fr], holdout_score=ens,
               runtime_sec=time.time() - t0,
               extra={"loc_alone": loc_alone, "kernel_only": k_only, "ens": ens,
                      "loc_weight": w.get("loc", 0.0), "resid_corr": corr, "calib_a": float(a)})
    exp.note(f"loc⊕kernel {ens:.3f} vs kernel {k_only:.3f} (w={w.get('loc',0):.3f}, corr={corr:.2f}, loc-alone {loc_alone:.3f})")
    exp.commit()

    dash.verdict(VER, ens, time.time() - t0, simple_avg=k_only, parent=9.155)
    v = ("✅ SUCCEED — locator adds" if ens < k_only - 0.05
         else "⚠ marginal" if ens < k_only - 0.02 else "✗ no lift → HARVEST")
    print(f"=== {VER}: {v} | loc⊕kernel {ens:.3f} vs kernel {k_only:.3f} | loc-alone {loc_alone:.3f} "
          f"| corr {corr:.2f} | {time.time()-t0:.0f}s ===")


if __name__ == "__main__":
    main()
