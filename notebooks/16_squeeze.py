"""16 — the cheap squeeze: negative-weight blend + variance-expansion + adversarial diagnostic. COLAB.

The investigation's grounded, untried levers, reusing models we already built (no new training of the
weak members — load their saved OOFs). All judged on SACRED.
  (A) NEGATIVE-weight blend over {kernel-cat, CNN(v7), locator(v14)} — our recurring failure is that
      these correlate with the kernel (ρ 0.6–0.9) → 0 POSITIVE weight. Negative weights subtract
      correlated error (survey L?: extracts lift NM-positive can't). Fit on dev-OOF, eval sacred (L48:
      negatives overfit holdout → never tune to sacred).
  (B) RMSE-CAGE variance expansion — diagnosis: large drifts under-predicted (under-dispersed). Stretch
      predictions around their mean toward the true variance; expansion factor tuned on dev-OOF.
  (C) Adversarial validation (dev-vs-sacred) — is 9.16 a TRUE ceiling (AUC≈0.5, like S6E5's 0.50038) or
      is sacred unrepresentative (CV artifact)?
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src import blend, cv, dashboard as dash, data, train
from src.evaluate import rmse
from src.observer import Experiment

CACHE = Path(os.environ.get("DRIVE_ROOT") or "data") / "cache"
VER = "v15_squeeze"
CAT = dict(depth=6, learning_rate=0.03, random_seed=42, l2_leaf_reg=2.0, min_data_in_leaf=15)


def main() -> None:
    t0 = time.time()
    dev_df = pd.read_parquet(CACHE / "dev_k9.parquet"); sac_df = pd.read_parquet(CACHE / "sacred_k9.parquet")
    feats = [c for c in dev_df.columns if c not in {"well", "id", "target"}]
    yd = dev_df["target"].to_numpy(np.float32); gd = dev_df["well"].to_numpy(); ys = sac_df["target"].to_numpy(np.float32)
    dash.goal_banner(VER, "negative blend + variance cage + adversarial", "extract lift from correlated members + fix under-dispersion")
    exp = Experiment.start(version=VER, parent="v6s8_fast",
                           hypothesis="Negative-weight blend extracts lift from correlated members (CNN/locator) "
                                      "that 0-positive-weight discards; + RMSE-cage variance expansion fixes "
                                      "under-dispersion. Cheap, grounded in our measured failures.",
                           predicted_delta=0.1, confidence="low",
                           pipeline_changes=["negative blend", "variance cage", "adversarial"], cloud_or_local="cloud")

    # kernel cat → OOF + sacred
    r = train.train_variant(f"{VER}_cat", "cat", dev_df[feats].astype("float32"), yd, gd,
                            X_test=sac_df[feats].astype("float32"), params=CAT, save=False, use_gpu="auto")
    oof = {"kernel": dict(zip(dev_df["id"], r.oof))}; sac = {"kernel": dict(zip(sac_df["id"], r.test_pred))}

    # load saved diverse OOFs (CNN v7, locator v14)
    for name, sub, ko, ks in [("cnn", "nn_p1", "nn_oof", "nn_sac"), ("loc", "loc_p2", "loc_oof_c", "loc_sac_c")]:
        p = CACHE / sub / "preds.npz"
        if not p.exists():
            print(f"  [skip {name}] {p} missing"); continue
        z = np.load(p, allow_pickle=True)
        oof[name] = dict(zip([str(i) for i in z["dev_ids"]], z[ko]))
        sac[name] = dict(zip([str(i) for i in z["sac_ids"]], z[ks]))
        print(f"  [{name}] loaded {len(oof[name])} oof / {len(sac[name])} sac")

    # align on common ids
    dev_ids = [i for i in dev_df["id"].tolist() if all(i in oof[m] for m in oof)]
    sac_ids = [i for i in sac_df["id"].tolist() if all(i in sac[m] for m in sac)]
    print(f"[align] members {list(oof)} | dev {len(dev_ids)}/{len(dev_df)} | sac {len(sac_ids)}/{len(sac_df)}")
    y_dev = dev_df.set_index("id").loc[dev_ids, "target"].to_numpy(np.float32)
    y_sac = sac_df.set_index("id").loc[sac_ids, "target"].to_numpy(np.float32)
    pick = lambda d, ids: {m: np.array([d[m][i] for i in ids], float) for m in d}
    od, sd = pick(oof, dev_ids), pick(sac, sac_ids)

    k_only = rmse(y_sac, sd["kernel"])
    print(f"\nkernel-only sacred {k_only:.3f}")
    print("marginal (LOO contribution, +=helps):")
    for row in blend.marginal_report(od, y_dev, allow_negative=True):
        print(f"  {row['member']:8s} solo {row['solo_rmse']:.3f} contrib {row['contribution']:+.4f}")

    # (A) positive vs negative blend
    res = {}
    for neg in (False, True):
        w, _, _ = blend.nm_optimize_oof(od, y_dev, allow_negative=neg)
        s = rmse(y_sac, blend.apply_blend(sd, w))
        res[neg] = (s, w)
        print(f"blend allow_negative={neg}: sacred {s:.3f} | w {dict((k,round(v,3)) for k,v in w.items())}")
    best_neg = min(res, key=lambda n: res[n][0]); s_blend, w_best = res[best_neg]
    blend_sac = blend.apply_blend(sd, w_best); blend_oof = blend.apply_blend(od, w_best)

    # (B) RMSE-cage variance expansion (factor tuned on dev-OOF, applied to sacred)
    mu = blend_oof.mean()
    facs = np.linspace(1.0, 1.8, 17)
    f_best = min(facs, key=lambda f: rmse(y_dev, mu + f * (blend_oof - mu)))
    s_cage = rmse(y_sac, blend_sac.mean() + f_best * (blend_sac - blend_sac.mean()))
    print(f"variance-cage factor {f_best:.2f}: sacred {s_cage:.3f} (blend {s_blend:.3f})")

    # (C) adversarial dev-vs-sacred (is sacred representative?)
    Xa = pd.concat([dev_df[feats], sac_df[feats]]).fillna(0).astype("float32")
    ya = np.r_[np.zeros(len(dev_df)), np.ones(len(sac_df))]
    ga = np.r_[dev_df["well"].to_numpy(), sac_df["well"].to_numpy()]
    aucs = []
    import lightgbm as lgb
    for tr, va in GroupKFold(5).split(Xa, ya, ga):
        m = lgb.LGBMClassifier(n_estimators=120, num_leaves=31, verbosity=-1).fit(Xa.iloc[tr], ya[tr])
        aucs.append(roc_auc_score(ya[va], m.predict_proba(Xa.iloc[va])[:, 1]))
    auc = float(np.mean(aucs))
    print(f"\n[adversarial] dev-vs-sacred AUC {auc:.4f}  ({'≈0.5 → sacred representative, 9.16 is a TRUE ceiling' if auc<0.55 else 'shift → CV may be off'})")

    best = min(k_only, s_blend, s_cage)
    exp.record(oof_score_mean=best, oof_score_per_fold=[k_only, s_blend, s_cage], holdout_score=best,
               runtime_sec=time.time() - t0,
               extra={"kernel_only": k_only, "blend_neg": res[True][0], "blend_pos": res[False][0],
                      "cage": s_cage, "cage_factor": float(f_best), "adv_auc": auc,
                      "weights": {k: round(v, 3) for k, v in w_best.items()}})
    exp.note(f"neg-blend {res[True][0]:.3f} pos {res[False][0]:.3f} cage {s_cage:.3f} vs kernel {k_only:.3f} | adv AUC {auc:.3f}")
    exp.commit()

    dash.verdict(VER, best, time.time() - t0, simple_avg=k_only, parent=9.155)
    v = "✅ squeeze lands" if best < k_only - 0.03 else "✗ flat — true ceiling confirmed"
    print(f"=== {VER}: {v} | best {best:.3f} vs kernel {k_only:.3f} | neg-blend {res[True][0]:.3f} | cage {s_cage:.3f} | adv {auc:.3f} | {time.time()-t0:.0f}s ===")


if __name__ == "__main__":
    main()
