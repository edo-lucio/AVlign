# FGW cross-modal alignment — multi-seed results interpretation

**Data**: `fgw_validation/results/results/fgw_n50_a*_s*_eval.json`  
**Scale**: 5 seeds × 7 α-values × 72 combos × 1 caption-agg/cost grid =
**2 520 evaluated configurations** (35 eval files, lite schema).  
**`n = 50`** sampled per side per combo.  
**Eval module**: `lite=True`, `B=200`, `null_K=50`, `ks=[1]`, item-level
structural bootstrap, baseline + null computed.  
**Witnesses present**: clip, clap, roberta, t5.
**`lex` (encoder-free) not in this run** — the raw caption files weren't
reachable from the eval host. The encoder-correlation defense is therefore
*not* fully closed by this data; re-running with the dataset folders
mounted will add the encoder-free witness.

---

## TL;DR

> **The pre-registered H_FGW test (primary config, `recall@1_t5` evaluated
> at α* tuned on `recall@1_clap`) decisively fails: Δ = −0.072 ± 0.027,
> z = −2.64.** This is a *statistically significant negative result*, not
> just an absence-of-positive — FGW with the pre-registered primary
> configuration is *worse than text-only retrieval beyond 2 standard errors
> across 5 seeds*.

The exploratory grid corroborates the negative headline at every
disaggregation: 168/7 560 cells (2.2 %) show any FGW > baseline win, mean
Δ = −0.11 to −0.14 per witness, and **no combo, witness, α, seed, or cost
convention recovers a defensible positive lift after multiplicity correction.**

What's new vs the prior single-seed analysis:

1. The pre-registered test now has a **proper z-statistic** (multi-seed SE);
   it lands clearly on the wrong side.
2. Structural fit **does** exceed the random-π null at α ≥ 0.25, but **fails
   to translate into semantic gain** at any α — the predicted dissociation
   from EXPERIMENT.md §11(5).
3. The geodesic ablation H_geo is **not supported**: `geo_cos` performs
   between `cos_cos` and `cos_neg`, with no meaningful improvement.
4. The dataset-overlap diagnostic is **brutal**: under both clip and t5
   witnesses, **only 5–7 % of Clotho audios are anyone's argmax** (Gini
   0.97–0.98). This is the asymmetric-coverage regime where balanced FGW
   structurally cannot win — and it explains the size of the negative.

---

## 1. The pre-registered primary test

EXPERIMENT.md §4.1 pre-registered the configuration
`(image=clip, audio=clap, bridge=clip, cost=cos_cos, agg=mean)` with α*
tuned on `recall@1_clap` and evaluated on `recall@1_t5`. Across 5 seeds:

| selection: `recall@1_clap` peaks at | α* = **0.25** (mean 0.136) |
|---|---|
| evaluation witness | `recall@1_t5` (held out from selection) |
| FGW `recall@1_t5`        | **0.072 ± 0.021** (mean ± across-seed SE) |
| baseline `recall@1_t5`   | **0.144 ± 0.012** |
| Δ                         | **−0.072 ± 0.027** |
| **z-statistic**           | **−2.64** |

The decision rule from §11 was Δ > 0 at z ≥ +2. The observed z is **−2.64
on the wrong side** — a clean rejection of H_FGW for the primary configuration.

This is not a "near-miss" or "weak negative." It is the *expected* outcome
under the alternative hypothesis "FGW is strictly worse than baseline in this
regime," which the rest of the analysis confirms.

### Primary config α-sweep (multi-seed)

| α | FGW recall@1_clap | baseline | Δ | FGW pearson | null_hi |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.112 ± 0.021 | 0.280 | −0.168 ± 0.032 | 0.009 | 0.110 |
| 0.10 | 0.128 ± 0.026 | 0.280 | −0.152 ± 0.039 | 0.063 | 0.110 |
| 0.25 | **0.136 ± 0.021** | 0.280 | −0.144 ± 0.038 | 0.148 | 0.110 |
| 0.50 | 0.120 ± 0.017 | 0.280 | −0.160 ± 0.041 | 0.279 | 0.110 |
| 0.75 | 0.100 ± 0.013 | 0.280 | −0.180 ± 0.047 | 0.377 | 0.110 |
| 0.90 | 0.064 ± 0.017 | 0.280 | −0.216 ± 0.040 | 0.413 | 0.110 |
| 1.00 | 0.016 ± 0.004 | 0.280 | −0.264 ± 0.041 | 0.446 | 0.110 |

**FGW never reaches baseline at any α.** The shape of the curve is
informative though: a shallow interior optimum at α = 0.25 (the structural
term starts contributing without yet swamping the W signal), then steady
decay as α grows. Pure-GW (α=1) collapses to ~chance recall (1/50 = 0.02).

---

## 2. Cell-by-cell exploratory grid

Across all 7 560 (combo × witness) cells:

| witness | n | mean Δ | std Δ | max Δ | % cells with FGW > baseline |
|---|---:|---:|---:|---:|---:|
| clip    | 1890 | −0.143 | 0.080 | +0.020 | 0.2 % |
| clap    | 1890 | −0.114 | 0.092 | +0.080 | 7.6 % |
| roberta | 1890 | −0.135 | 0.073 |  0.000 | 0.0 % |
| t5      | 1890 | −0.110 | 0.072 | +0.020 | 1.2 % |
| **all** | 7560 | **−0.125** | **0.080** | **+0.080** | **2.2 %** |

Counts of "wins" over the grid by chance under H_0 (FGW = baseline at each
cell, 50/50 better/worse): even with no multiplicity correction, the
expected wins under the null is 3 780. Observed is 168 — **22× lower than
chance** in the wrong direction. Even a generous BH correction over the grid
yields **q = 1.0 for every claimed positive**: the apparent +0.08 win on
`(clip × clap × roberta × cos_neg × mean, α=0.9, seed=3, witness=clap)` does
not survive multiplicity correction because hundreds of comparable cells at
the same nominal level produce no positive signal.

### Bridge × α heatmap (Δ recall@1_avg, multi-seed mean)

| bridge   | α=0.00 | α=0.10 | α=0.25 | α=0.50 | α=0.75 | α=0.90 | α=1.00 |
|---|---:|---:|---:|---:|---:|---:|---:|
| clap     | −0.114 | −0.114 | −0.115 | −0.115 | −0.114 | −0.114 | −0.161 |
| clip     | −0.133 | −0.133 | −0.134 | −0.134 | −0.136 | −0.142 | −0.193 |
| roberta  | −0.104 | −0.107 | −0.115 | −0.119 | −0.124 | −0.129 | −0.175 |
| t5       | −0.105 | −0.104 | −0.104 | −0.106 | −0.108 | −0.114 | −0.145 |

Smallest gap: t5-bridge at α = 0.10–0.25, Δ ≈ −0.10. Largest gap:
clip-bridge at α = 1.0, Δ = −0.19. **No bridge × α cell beats baseline.**

---

## 3. Where FGW *does* show real signal — and why it doesn't help

### Structural fit exceeds the random-π null at α ≥ 0.25

| α | FGW pearson (avg) | null_pearson_hi | exceeds null? |
|---:|---:|---:|---|
| 0.00 | 0.034 | 0.087 | no |
| 0.10 | 0.060 | 0.087 | no |
| 0.25 | 0.087 | 0.087 | **borderline** |
| 0.50 | 0.127 | 0.087 | **YES** |
| 0.75 | 0.182 | 0.087 | **YES** |
| 0.90 | 0.243 | 0.087 | **YES** |
| 1.00 | 0.346 | 0.087 | **YES (4× null)** |

The FGW structural correlation is genuinely informative — a randomly chosen
permutation gets `pearson_dist_corr ∈ [-0.05, 0.087]` (95 % null interval),
while FGW at α ≥ 0.5 reliably scores 2-4× above that ceiling. **The GW term
is doing real geometric work.**

The catch: this geometric work **does not translate into semantic recall**.
At α=1 (pure GW, maximal structural fit), `recall@1_avg` collapses to
chance. FGW finds permutations that preserve intra-modal geometry but
*pair semantically unrelated items*. This is exactly the dissociation
predicted in EXPERIMENT.md §11(5): "if structural is high but semantic is
low everywhere, FGW is solving its own objective without producing a
meaningful matching." The data shows it cleanly.

### Triplet agreement: small lift at high α

| α | FGW triplet | baseline | Δ |
|---:|---:|---:|---:|
| 0.00 | 0.511 | 0.531 | −0.020 |
| 0.50 | 0.530 | 0.531 | −0.001 |
| 1.00 | 0.576 | 0.531 | **+0.045** |

A genuine but small win at α=1 (~5pp above chance and above the row-wise
greedy baseline). This is consistent with the structural-fit story: FGW's
GW term recovers ordinal NN structure to a real but limited degree.

---

## 4. The geodesic ablation (H_geo) is not supported

`geo_cos` is gated to (CLIP image, CLAP audio) only — the lone hyperspherical
encoder pair. Across 280 records per cost convention on this pair:

| cost convention | recall@1_avg | pearson | triplet |
|---|---:|---:|---:|
| `cos_cos` | 0.0569 | 0.252 | 0.557 |
| `cos_neg` | 0.0656 | 0.154 | 0.538 |
| `geo_cos` | 0.0620 | 0.165 | 0.537 |

`geo_cos` lands **between** `cos_cos` and `cos_neg` on every metric — no
meaningful improvement over the chord (`cos_cos`) baseline. The
square_loss landscape does shift (`fgw_dist` differs), but the resulting
permutations are no better. **H_geo: not supported.** The reviewer's
prediction (rank-based metrics are invariant to monotone rescalings, so
geodesic distance can only matter through `square_loss` interactions) is
borne out — the interactions are not strong enough to matter at this n.

---

## 5. Dataset overlap diagnostic — the structural cause

The new `dataset_overlap.py` reveals the regime mismatch directly. Per-image
best-match similarity and per-audio coverage asymmetry, computed on the
full Flickr8k (test) × Clotho (development) cross product:

| witness | median max-sim | q25 — q75 | **audios used** | Gini |
|---|---:|---|---:|---:|
| `clip`  | 0.813 | 0.768 — 0.854 | **7.4 %** | 0.967 |
| `t5`    | 0.872 | 0.859 — 0.885 | **5.5 %** | 0.980 |

Read off the right-hand columns: **only 5–7 % of Clotho audios are anyone's
top match.** 93–95 % of the audio side is dead inventory — no Flickr8k
image's captions point to it as the best match under either witness. The
Gini coefficient of the per-audio argmax-count distribution is 0.97–0.98
(0 = uniform, 1 = monopoly) — extreme concentration on a few "hub" audios.

This is the asymmetric-coverage regime where **balanced FGW (uniform-marginal
LP-assignment, permutation T) is structurally the wrong tool**:

- Row-wise greedy baseline: 100 images can all map to the 5 best-matching
  audios, getting 100 correct matches.
- Balanced FGW: forced bijection. The 5 good audios can absorb at most 5
  images. The other 95 images are sent to *bad* matches because the
  bijection constraint forbids reuse.

The size of the gap (Δ ≈ −0.12) is consistent with the size of the
asymmetry: roughly 10–15 % of images "deserve" one of a small set of hub
audios, and the LP-assignment penalty dominates exactly there. Numerically:
baseline `recall@1_clap = 0.280` ≈ images-with-hub-match / total; FGW
`recall@1_clap ≈ 0.13` ≈ small-fraction-of-bijection-getting-right-answer.

---

## 6. The α = 0 "sanity check" remains fail-by-design

EXPERIMENT.md §7 originally claimed FGW(α=0) should equal baseline. The new
data confirms again: not equal, by ~0.10–0.27 in `recall@1` per seed. As
explained in the prior `RESULTS.md`, this is **structural** — FGW(α=0)
solves the LP-assignment problem (permutation T, every column used once),
while baseline does row-wise greedy retrieval (collisions allowed). Two
different problems by design.

**Implication unchanged from before.** EXPERIMENT.md §7 should describe the
right α=0 calibration as: *"FGW(α=0) should match a Hungarian-on-M baseline,
not row-wise argmax."* That separate baseline is not yet implemented — when
it is, the calibration becomes informative again.

---

## 7. Differential predictions: what the ablation axes show

| axis | finding |
|---|---|
| **caption agg** | `mean` > `first` by Δ = +0.020 — mild positive ablation. |
| **cost convention** | cos_cos / cos_neg / geo_cos all within 0.007 — no preferred convention. |
| **image encoder** | clip vs dinov2 within 0.0004 — no effect. |
| **audio encoder** | clap vs ast within 0.0002 — no effect. |
| **bridge encoder** | t5 ≥ clap > roberta > clip on FGW Δ (smallest gap at t5). Inverse of the baseline ranking (clip > clap > roberta > t5 on baseline). |

The bridge ranking inversion is the same finding as the prior single-seed
analysis: stronger multimodal bridges (clip, clap) help baseline more than
they help FGW, so FGW's gap to baseline is *largest* with the strongest
bridges. Mechanistically: FGW's GW term overrides the bridge's signal, and
overrides it more destructively when the bridge had more signal to begin
with.

The flatness of image / audio / cost dimensions is informative: when an
algorithm fails uniformly across encoder choices, encoder differences
disappear in the noise. This is consistent with reading (9a) from the
prior analysis: the regime is the bottleneck, not the encoder choice.

---

## 8. Transport plan diagnostics — solver healthy

Across all 2 520 combos:

| metric | value | reading |
|---|---:|---|
| `entropy_norm`     | 0.500 | exact value of `log(n)/log(n²)` for permutation T |
| `top1_mass`        | 1.000 | every row puts all mass on its argmax |
| `coverage`         | 1.000 | every column is used exactly once |
| `mutual_best_rate` | 1.000 | T is a permutation matrix |

Identical to the prior single-seed analysis. **The FGW solver is fine.** It
finds clean permutations at every α. The matchings it produces are simply
the wrong matchings for asymmetric distributions.

---

## 9. What this all adds up to

The two strongest findings in this multi-seed run:

1. **Pre-registered H_FGW test fails at z = −2.64** — the cleanest
   falsification possible without multi-seed SE was already strong from
   the unfiltered single-seed grid; with multi-seed, it now meets the
   "negative result with statistical confidence" bar.

2. **The asymmetric-coverage diagnostic gives a principled mechanism**:
   only 5–7 % of Clotho audios are anyone's top match, Gini 0.97–0.98.
   Balanced FGW *cannot* exploit this asymmetry and pays the
   LP-assignment penalty exactly where it hurts.

These two findings together support reading (9a) from the prior analysis:
**the regime, not FGW, is the failure point.** Balanced FGW with uniform
marginals is structurally wrong for these distribution pairs. The natural
fix — unbalanced FGW, which permits asymmetric mass — was not tested here
but is now well-motivated.

---

## 10. Recommended next actions

In priority order:

1. **Add the encoder-free `lex` witness.** This run was missing it (raw
   captions not reachable). Re-run eval *only* (no need to redo FGW) with
   the dataset folders accessible. Expected impact: confirms the negative
   result is not encoder-leakage — already extremely strong evidence
   from the 4 encoder witnesses, but lex closes the loop.

2. **Run the **filtered** sweep** (using `jobs/fgw_build_filter.job` then
   `FILTER_INDICES=...`). The MNN-filtered subset gives balanced FGW a
   fair shot. The smoke test at n=50 already showed this regime change
   bumps absolute baseline `recall@1` from 0.13 → 0.38 — i.e. the ceiling
   rises dramatically, and FGW gets some of that lift but still loses by
   a similar absolute margin (~0.10) to baseline on independent witnesses.
   The full multi-seed filtered run will tell whether *that* gap closes
   under symmetric coverage.

3. **Try unbalanced FGW** (`ot.unbalanced.fused_unbalanced_gromov_wasserstein`
   or `ot.partial.partial_fused_gromov_wasserstein`). With KL-divergence
   penalties on marginal residuals instead of hard uniform constraints,
   multiple images can map to the same hub audio — the matching can
   replicate what row-wise greedy is doing, while also using the GW term.
   This is the principled fix to the LP-assignment problem and the most
   promising path to a positive result on the unfiltered data.

4. **Otherwise: write up the negative result honestly.** The infrastructure
   is now fully publishable: pre-registered test, multi-seed SE,
   item-level bootstrap, structural null, encoder-free witness (once
   added), dataset-overlap diagnostic with coverage-asymmetry. A clean
   "we built a falsifiable evaluation pipeline and balanced FGW failed it,
   here's the principled mechanism via asymmetric coverage, and here's
   the unbalanced-FGW alternative we propose for future work" is a more
   defensible contribution than any contrived positive on this regime
   would be.
