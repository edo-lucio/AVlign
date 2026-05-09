# FGW cross-modal alignment — experiment specification

This document describes the experiment implemented under `fgw_validation/`: aligning
**unpaired** image and audio embeddings with **Fused Gromov–Wasserstein (FGW)**, using
text captions as a bridge, and evaluating the resulting transport plan in the absence
of ground-truth pairs.

```
fgw_validation/
├── encode.py            produce per-encoder embeddings
├── fgw_text_bridge.py   run the FGW alignment grid
├── eval.py              score every transport plan (this doc focuses here)
├── cka.py               pairwise CKA across encoders (representation similarity)
├── plots.py             render figures from the eval JSONs
└── data/                inputs (datasets, embeddings) and outputs (results, figures)
```

---

## 1. The question and what it actually tests

Two text-described modalities — **images** (Flickr8k) and **audio clips** (Clotho) —
have no observed pairing: image *i* in Flickr8k has no "correct" audio match in
Clotho. Each item, however, has 5 free-form text captions describing it.

> **Given paired (item, caption) annotations on both sides — but *no* observed
> (image, audio) pairs — can we recover a meaningful image ↔ audio matching from
> intra-modal geometry (image-image distances, audio-audio distances) plus a
> *text-bridge* (caption-caption similarities)?**

This is **not** "alignment without paired data" in the strong sense. Captions are
paired text annotations on each side, and the bridge term `M` directly exploits them.
The narrower regime tested here is: *no observed (image, audio) pairs, but per-item
caption supervision on both sides.* The strong "no captions either" use-case (where
FGW would transfer alignment from a small reference-aligned subset to a captionless
remainder) is **not** tested by this design.

If FGW is doing something useful in this narrower regime, the transport plan should:

1. preserve **structural** relationships (close images map to acoustically close audio),
2. agree with **semantic** judgements made by *held-out* witnesses (text encoders
   not used as the FGW bridge, plus an encoder-free lexical witness — see §6),
3. **outperform a trivial baseline** that just matches captions in the bridge encoder
   directly, with no FGW. **This is the load-bearing test.** If the GW term
   contributes nothing beyond what bridge-text retrieval already does, FGW has no
   value-add in this regime.

(3) is what the eval pipeline is specifically constructed to make falsifiable, via
the `baseline_*` columns and bootstrap CIs.

---

## 2. The FGW formulation

For a single combination — pick (`image_encoder`, `audio_encoder`, `text_encoder`,
`cost_convention`, `caption_agg`) — we sample `n` images and `n` audio clips uniformly
at random, then form three cost matrices:

| Symbol | Shape | Meaning |
|---|---|---|
| `C_i` | (n, n) | intra-image distances under the chosen cost convention |
| `C_a` | (n, n) | intra-audio distances under the chosen cost convention |
| `M`   | (n, n) | image-text → audio-text feature cost via the bridge encoder's caption embeddings |

We solve, with `α ∈ [0, 1]` controlling the GW vs feature blend:

$$
T^{*} \;=\; \arg\min_{T \in \Pi(\mathbf{1}/n,\,\mathbf{1}/n)}
\;(1-\alpha)\,\langle T,\,M\rangle
\;+\;
\alpha\!\!\sum_{i,j,k,l}\!\! \big(C_i[i,k] - C_a[j,l]\big)^{2}\, T[i,j]\, T[k,l].
$$

This is `ot.gromov.fused_gromov_wasserstein` with `loss_fun="square_loss"`
(`fgw_text_bridge.py:184`). The two endpoints are interpretable:

- **α = 0**: pure Wasserstein on `M` only. FGW reduces to **caption-to-caption retrieval
  in the bridge encoder** — equivalent (modulo entropic smoothing) to the text-only
  baseline used by the eval (see §6.1).
- **α = 1**: pure Gromov–Wasserstein. FGW receives **no text information** and aligns
  purely on intra-modal geometry. This is the cleanest "is shape alone enough?" test.

Intermediate α blends the two signals.

The hard match used downstream is `π(i) = argmax_j T[i, j]`.

---

## 3. Cost conventions (`fgw_text_bridge.py:111-153`)

Three choices for how to populate `C_i`, `C_a`, `M`:

| convention | `C` (intra-modal) | `M` (cross-modal feature) | scope |
|---|---|---|---|
| `cos_cos` | `1 − cos(z, z)` ∈ [0, 2] (chord) | `1 − cos(T_i, T_a)` ∈ [0, 2] | all combos |
| `cos_neg` | `1 − cos(z, z)` ∈ [0, 2] | `−⟨T_i, T_a⟩` (sign-flipped raw inner product) | all combos |
| `geo_cos` | `arccos(cos)/π` ∈ [0, 1] (geodesic, rescaled for parity) | `1 − cos(T_i, T_a)` ∈ [0, 2] | only `(clip, clap)` |

`geo_cos` is **gated** by `_combo_is_valid` to encoder pairs whose source spaces are
both contrastively-trained L2-normalised hyperspheres — i.e. only **CLIP image + CLAP
audio**. DINOv2 (self-distillation, no L2 norm guarantee) and AST (supervised classifier
logits, not InfoNCE) do not occupy a meaningful hypersphere, so geodesic distance is
not a principled cost there.

> **Why three conventions matter.**
> `cos_cos` is the standard choice. `cos_neg` keeps `M` on a different scale than `C` —
> probes whether FGW is sensitive to the relative magnitudes of the GW and W terms.
> `geo_cos` tests whether the **chord-vs-arc distinction** matters when both source
> spaces literally live on `S^{d−1}` (Kornblith-style argument: arccos is the proper
> Riemannian distance on the hypersphere; chord is its Euclidean approximation in the
> ambient space).

---

## 4. Ablation grid

Per α-value:

| axis | values | size |
|---|---|---|
| image encoder       | `clip` (small), `dinov2` (small)         | 2 |
| audio encoder       | `clap` (medium), `ast` (medium)          | 2 |
| text encoder (bridge)| `clip`, `clap`, `roberta`, `t5`         | 4 |
| caption aggregation | `mean` (over 5 captions), `first` (caption 0) | 2 |
| cost convention     | `cos_cos`, `cos_neg`, `geo_cos` (gated)  | 3 (with gating) |

```
cos_cos:  2 × 2 × 4 × 2  =  32
cos_neg:  2 × 2 × 4 × 2  =  32
geo_cos:  1 × 1 × 4 × 2  =   8        ← (clip, clap) only
                          ─────
                          = 72 combos / α
```

Default α-sweep: `{0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}` × seed 0 → **504 FGW alignments**
per full sweep run. n=200 by default (`fgw_text_bridge.py:237`).

### Sub-hypotheses tested by these axes

- **H_α**: There is a non-trivial interior optimum α* ∈ (0, 1). If α*=0, the GW term
  adds nothing; if α*=1, the bridge adds nothing.
- **H_bridge**: Multimodal bridges (CLIP, CLAP — trained with paired text supervision)
  produce stronger `M` than text-only bridges (RoBERTa, T5).
- **H_image / H_audio**: CLIP outperforms DINOv2 for image-side `C_i` because its space
  is text-aware. CLAP outperforms AST for the same reason on audio.
- **H_geo**: For the (CLIP, CLAP) pair, `geo_cos` outperforms `cos_cos` because the
  underlying manifold is a hypersphere and the Riemannian distance is the principled
  one. (Caveat: chord, cosine, and geodesic are monotone transforms of each other, so
  rank-based metrics are *invariant* to this choice; only `square_loss`-driven optima
  can differ.)
- **H_caption_agg**: Mean-pool over 5 captions is a stronger signal than the first
  caption alone. (Likely true; weak baseline.)

### Foundational hypothesis (the load-bearing one)

- **H_FGW**: For the **pre-registered primary configuration** (§4.1), there exists
  α* ∈ (0, 1) such that

$$
\textsf{recall}@1_{\text{FGW},\,\alpha^*} \;>\; \textsf{recall}@1_{\text{baseline}}
\quad\text{across multi-seed runs (across-seed SE, not within-seed bootstrap)},
$$

  on the pre-registered held-out witness. **If this fails on the primary
  configuration, FGW does not add value beyond bridge-text retrieval in this
  regime**, and any positive results elsewhere in the grid must be treated as
  exploratory and reported with multiplicity correction.

### 4.1 Pre-registered primary configuration

To control for the multiple-comparisons problem inherent in a 504-cell ablation
grid, **one** configuration is pre-registered as the primary H_FGW test. All other
combinations are exploratory and reported with Benjamini–Hochberg q-values
(see §6.4).

| component | choice | a-priori rationale |
|---|---|---|
| image encoder       | `clip` (small)    | L2-normalised contrastive hypersphere; shares InfoNCE objective with `clap` |
| audio encoder       | `clap` (medium)   | Same hypersphere geometry; only audio encoder with text-aligned training |
| bridge text encoder | `clip` (small)    | Contrastive image-text training → tightly bound `M` for the image side |
| cost convention     | `cos_cos`         | Standard chord distance; conservative default |
| caption aggregation | `mean`            | Averages over 5 captions; stronger signal than `first` |
| α*                  | tuned on `recall@1_clap`, evaluated on `recall@1_t5` | two-stage selection prevents optional-stopping bias |
| n                   | 200               | sweep default |
| seeds               | 5 (≥)             | for across-seed SE, not just within-seed bootstrap |

The selection encoder (`clap`) and the evaluation encoder (`t5`) are deliberately
disjoint: tuning α* on one witness and reporting H_FGW on a different one closes
the optional-stopping loophole. T5 is a text-only encoder with no contrastive
image-text or audio-text training, making it the cleanest available witness
(though still web-text correlated with the bridge — see §7 on encoder leakage).

**Pre-registration timestamp**: this configuration was committed to
`EXPERIMENT.md` *before* multi-seed runs were executed. The git log of this file
is the audit trail.

---

## 5. Data and embeddings

| dataset    | role           | items per split (used) | text captions / item |
|---|---|---|---|
| Flickr8k   | image source   | `test` split (default `train`) | 5 |
| Clotho     | audio source   | `development` split            | 5 |

Encoded embeddings live at:

```
data/embeddings/<dataset>/<split>/<encoder>_<size>_<modality>.pt
```

Each `.pt` is `dict(emb, ids, encoder, modality, dim)`:

- image / audio: `emb` is `(N, D)` after pooling.
- text: `emb` is `(N, 5, D)` (one row per caption); aggregation happens at FGW build
  time according to `--caption_aggs`.

Encoder sizes (`eval.py:50`): `clip-small`, `dinov2-small`, `clap-medium`,
`ast-medium`, `roberta-small`, `t5-small`.

---

## 6. Evaluation metrics

For every combo we compute **two parallel sets** of metrics — one for FGW's matching
`π_FGW(i) = argmax_j T[i,j]`, one for a **text-only retrieval baseline**:

$$
\pi_{\text{baseline}}(i) \;=\; \arg\max_j \; \cos\big(\, e_{\text{bridge}}(\text{img\_caps}_i),\; e_{\text{bridge}}(\text{aud\_caps}_j) \,\big)
$$

i.e. matching captions of image `i` against captions of audio `j` in the bridge
encoder, with **no FGW at all**. Baseline metrics are stored with a `baseline_` prefix.

**Reading FGW − baseline** is the actual "did FGW help" comparison.

### 6.1 Three metric families

#### (A) Semantic validity — held-out witnesses (`eval.py:_semantic`)

For each witness `e` in the witness set:

- **Encoder witnesses**: `e ∈ {clip, clap, roberta, t5} \ {bridge}` — embed image-captions
  and audio-captions in `e`, mean-pool over 5 captions, form `sim[i,j] = cos(...)`.
- **Encoder-free witness** `e = lex`: tokenise the 5 captions per item (lowercased,
  stop-words and short tokens stripped), take the union as the item's token set, and
  define `sim[i,j] = |toks_i ∩ toks_j| / |toks_i ∪ toks_j|` (Jaccard). This witness
  is **fully outside the encoder leakage chain** — it shares no training data with
  any encoder in the grid (see §7 for why this matters).

Then for each predicted match `π(i)`:

| metric | definition |
|---|---|
| `caption_sim_mean_<e>`     | `mean_i sim[i, π(i)]` — semantic closeness in `e`'s view |
| `caption_sim_random_<e>`   | same with one random permutation `π_rand` (chance baseline) |
| `caption_sim_lift_<e>`     | mean − random — signal above chance |
| `recall@{1,5,10}_<e>`      | fraction of `i` where `π(i)` is in top-k of row `i` of `sim` |
| `soft_recall@{1,5,10}_<e>` | `Σ_{i,j} T[i,j] · 1[j ∈ topk(i)]` — T-weighted recall (FGW only; baseline has no `T`) |
| `mean_rank_<e>`            | mean rank of `π(i)` in row `i` (1-indexed, lower is better) |
| `median_rank_<e>`          | median rank — robust to right tail (n=200 has heavy upper tail) |
| `mrr_<e>`                  | mean reciprocal rank `mean_i 1 / rank_pos[i]` |

The **bridge encoder is excluded** from the encoder-witness list — otherwise we'd
be grading FGW against the same encoder it explicitly used in `M`, which would be
circular (see §7). The lexical witness is always included since it shares nothing
with the bridge.

#### (B) Structural / geometric alignment (`eval.py:_structural`, `_triplet_agreement`)

Encoder-independent — does π preserve pairwise distances?

| metric | definition |
|---|---|
| `pearson_dist_corr`     | Pearson `ρ(C_i[i, j], C_a[π(i), π(j)])` over all upper-triangle pairs |
| `spearman_dist_corr`    | rank version of the above |
| `triplet_agreement`     | for `n_triplets=2000` random `(a, b, c)`: fraction where "`b` closer to `a` than `c`" agrees between image and audio sides |
| `null_pearson_{mean,lo,hi}`  | Pearson under K=200 random permutations of π — distribution under structural null |
| `null_spearman_{mean,lo,hi}` | same, rank version |

**Reading structural metrics correctly.** `pearson_dist_corr` is essentially the GW
objective evaluated on the FGW solution — high values say "the optimizer converged,"
not "the alignment is meaningful." The `null_*` columns give the structural-null
reference: a structural metric is only informative if it lies **outside the null
percentile interval**. Note that the text-only `baseline_*` is *not* the right
comparison for structural metrics — the baseline ignores `C_i, C_a` by construction
and necessarily scores low — so the random-permutation null is what to compare
against. `triplet_agreement` is rank-invariant and slightly less tautological. See
§7 for the full circularity discussion.

**Bootstrap.** The structural-correlation CIs use **item-level resampling** (resample
the `n` images i with replacement, recompute the upper-triangle correlation on the
resampled sub-matrix), not pair-of-pairs. Each item participates in `n−1` pairs, so
pair-level resampling violates independence and produces anticonservative CIs.

#### (C) Transport-plan quality (`eval.py:_transport_stats`)

Properties of `T` itself, no embeddings touched:

| metric | definition |
|---|---|
| `entropy_norm`     | `H(T) / log(n_i · n_a)` ∈ [0, 1] (0 = permutation, 1 = uniform) |
| `top1_mass`        | `mean_i n_i · max_j T[i, j]` (1 = hard match) |
| `mutual_best_rate` | fraction of `i` with `argmax_i' T[i', π(i)] = i` (bidirectional matches) |
| `coverage`         | `|{π(i) : i}| / n_a` (1 = every audio target gets used) |

Sanity checks on the optimizer. **Not directly informative about alignment quality** —
`top1_mass` near 1 is necessary for the other metrics to be meaningful (otherwise
`argmax_j T[i, j]` is noise) but not sufficient.

### 6.2 Bootstrap confidence intervals

Every metric in (A) and (B) carries `<metric>_lo` and `<metric>_hi` fields, computed
as a percentile bootstrap (default `B=1000` resamples, 95% CI) over the per-image,
per-pair, or per-triplet contributions (`eval.py:_bootstrap_mean_ci`,
`_bootstrap_corr_ci`).

- Mean-bootstrap (recall@k, caption_sim_*, mean_rank, triplet_agreement): resample
  per-element vector with replacement, recompute mean, take percentile interval.
- Correlation-bootstrap (pearson, spearman): resample paired distances with
  replacement, recompute Pearson on the resample (for spearman, rank once and Pearson
  on ranks). Chunked over B to bound memory.

This is the only way to tell whether a 0.03 difference between two combos in
`recall@1` is signal or noise. With `n=200`, the binomial std on `recall@1` is
≈0.035 at p=0.5 — i.e. larger than most leaderboard differences without CIs.

### 6.3 Multi-seed aggregation (the headline reporting unit)

A single seed gives only **within-sample** uncertainty (which 200 items happened
to be drawn). To capture the variance over *which subset* of n items was sampled,
the sweep is run for **multiple seeds** (default ≥5 in the cluster job). For
every metric the headline figure is:

```
metric_mean = mean over seeds
metric_se   = std-over-seeds / sqrt(n_seeds)
```

Reported as `metric_mean ± metric_se` in plots and the leaderboard. The
within-seed bootstrap CIs (`_lo`, `_hi`) are still computed and stored per file —
they remain useful for diagnosing whether a single seed's signal is bootstrap-stable
— but the headline test for **H_FGW** uses across-seed SE, not bootstrap CIs.

### 6.4 Multiplicity correction

The exploratory leaderboard contains 504 (combo × α) cells × ~10 reported
metrics × 4 witnesses ≈ 20,000 hypothesis tests. A naive "FGW > baseline at 95%
CI" criterion has FWER ≈ 1 under the null.

For exploratory analysis, **Benjamini–Hochberg q-values** are computed across the
grid (one BH pass per metric × witness combination), and the leaderboard is
annotated with each cell's q-value. Cells with q < 0.05 are flagged as
exploratory hits worth following up; cells with q > 0.05 are explicitly *not*
considered evidence of FGW improvement.

The pre-registered primary configuration (§4.1) is exempted from BH correction —
it's a single pre-specified test, not a search.

### 6.5 What the eval does NOT do (yet)

- **Multi-permutation random baseline for `caption_sim_random_<e>`.**
  Currently uses one random π; noisy. Could average over ~100 permutations or use
  the analytic mean of off-diagonal `sim`.
- **Max-over-captions semantic witness.** Mean-pool destroys per-caption variation.
  Max-over-(k_img, k_aud) of `cos(cap_k_img, cap_k_aud)` is often a stronger signal.
- **Cross-validated alignment.** `idx_i, idx_a` are the same at FGW time and eval
  time (in-sample). Non-parametric OT does not have a clean train/test split, but
  this is worth saying explicitly. The multi-seed protocol partially addresses this
  by varying which items are sampled, but is not equivalent to held-out evaluation.

---

## 7. Circularity audit

A natural concern: if FGW uses cosine distances on encoder embeddings to compute the
plan, and eval uses cosine similarities on the same embeddings to grade it, are we
just asking the optimizer how well it optimized?

| metric family | circularity | why |
|---|---|---|
| Transport stats              | none | `T` is the only input |
| Structural (pearson/spearman) | **strong** | this *is* the GW objective, in disguise |
| Triplet agreement            | mild | rank-invariant but still self-graded on `C` |
| Semantic — held-out encoder  | weak | encoders share training data → soft leakage |
| Semantic — bridge encoder    | avoided | excluded by `_semantic` |

### Why the baseline (§6) is the firewall

Both metrics get re-computed under `π_baseline`. Reading **FGW − baseline** controls for:

- **Layer 2 circularity** (structural ≈ GW objective): if FGW's `pearson_dist_corr`
  is no higher than `baseline_pearson_dist_corr`, the structural fit was achievable
  by random caption matching, so FGW's GW term contributed nothing.
- **Layer 3 leakage** (held-out encoders are correlated with bridge): the same
  correlated-but-not-identical encoder is grading both FGW and the baseline, so any
  uniform bias cancels.

### α = 0 sanity check (calibration, not plausibility)

At α = 0, FGW minimises `⟨T, M⟩` only. The implementation uses POT's
**non-entropic** `ot.gromov.fused_gromov_wasserstein` (`fgw_text_bridge.py:184`),
which solves the LP exactly via Block Coordinate Descent. So at α=0:

$$
\pi_{\text{FGW}}(\alpha=0) \;=\; \pi_{\text{baseline}}\quad\text{(up to BCD numerical tolerance)}.
$$

This is a **calibration** — numerical equality is expected, not just rough
agreement. Concretely, **FGW metrics at α=0 should match baseline metrics
exactly** (modulo float roundoff). If they don't on a real run, there is a bug in
`M` construction or in the baseline path. If they do, the baseline implementation
is verified faithful.

(With the entropic variant `entropic_fused_gromov_wasserstein`, this would
degrade to a plausibility check because Sinkhorn smoothing adds an ε-term not
present in the baseline. We don't use that variant.)

---

## 8. Cross-encoder representation similarity (CKA)

Independent of the FGW alignment itself, `cka.py` computes pairwise **Centered Kernel
Alignment** (Kornblith et al. 2019) across all encoders for a fixed `(dataset, split,
modality)`:

$$
\mathrm{CKA}(X, Y) \;=\; \frac{\|Y_c^{\!\top} X_c\|_F^{2}}{\|X_c^{\!\top} X_c\|_F\,\|Y_c^{\!\top} Y_c\|_F}
\quad\text{(linear, closed form)}
$$

(also available with an RBF kernel via biased HSIC).

Two roles:

1. **Diagnostic**: if image-CLIP and image-DINOv2 have CKA ≈ 1, swapping them in the
   ablation grid shouldn't change anything; if CKA is low, differences in FGW outcome
   are evidence that the encoder choice carries real information.
2. **Performance vs CKA scatters** (`plots.py: plot_perf_vs_cka`): does FGW improve as
   the bridge text encoder moves closer (in CKA) to the image-side or audio-side
   encoders? Tests whether representation alignment between bridge and source spaces
   predicts FGW success.

---

## 9. Outputs and figures

### Files written

| path | producer | what |
|---|---|---|
| `data/results/fgw_*.json`            | `fgw_text_bridge.py` | per-α bundle of 72 combos with `T` summaries |
| `data/results/fgw_*_plans/<combo>.npz` | `fgw_text_bridge.py --save_plans` | full `T`, `idx_i`, `idx_a` per combo |
| `data/results/fgw_*_eval.json`       | `eval.py` | combos enriched with metrics + baselines + CIs |
| `data/cka_<dataset>_<split>_<modality>_<kernel>.json` | `cka.py` | CKA matrix |
| `data/figures/*.png`                 | `plots.py` | figures (see below) |

### Figure families (`plots.py`)

| section | what it shows |
|---|---|
| `alpha`        | metric vs α curves, faceted by bridge text encoder; lines per (image, audio) pair |
| `leaderboard`  | (image × audio) heatmap per (metric, bridge) at best-α |
| `transport`    | boxplots of entropy / top1_mass / coverage vs α |
| `cka`          | heatmaps of pairwise CKA |
| `umap`         | 2D embeddings of representation spaces (optional, requires `umap-learn`) |
| `perfcka`      | scatter of recall vs cross-encoder CKA |
| `geo`          | (CLIP, CLAP) only — metric vs α with one line per cost convention; tests `H_geo` |

### Currently NOT in the figures

The new `*_lo`, `*_hi`, and `baseline_*` fields produced by the updated `eval.py` are
present in the JSONs but **not yet rendered as error bars / baseline reference lines**
in the existing plots. Adding them is a small extension to `plots.py`.

---

## 10. How to run

**Local smoke test** (1 combo, n=30, ~5 s):

```bash
python -m fgw_validation.fgw_text_bridge \
    --n 30 --alpha 0.5 --seed 0 \
    --emb_dir fgw_validation/data/embeddings \
    --image_encoders clip --audio_encoders clap \
    --text_encoders clip --cost_conventions cos_cos \
    --caption_aggs mean \
    --save_plans --out /tmp/fgw_smoke.json

python -m fgw_validation.eval \
    --results /tmp/fgw_smoke.json \
    --emb fgw_validation/data/embeddings \
    --bootstrap_B 200
```

**Cluster — full sweep:**

```bash
EMB_DIR=data/embeddings bash jobs/fgw_alpha_sweep.sh 200 0   # 7 α × 72 combos
EMB_DIR=data/embeddings sbatch jobs/fgw_eval.job             # eval + CKA + plots
```

Outputs land in `fgw_validation/data/{results,figures,cka_*}` — override the figure
location with `FIG_DIR=...`. The eval enriches each `fgw_*.json` in-place as
`fgw_*_eval.json` and `plots.py` consumes those.

---

## 11. What counts as a positive result

Read the figures and leaderboard with this checklist:

### Primary test (single, pre-registered)

1. **For the §4.1 primary configuration**, on the pre-registered held-out witness
   `recall@1_t5`, at the α* selected by tuning on `recall@1_clap`:

$$
\textsf{recall}@1_{\text{FGW}} \;-\; \textsf{recall}@1_{\text{baseline}} \;>\; 0
\quad\text{at}\;\;\geq 2\;\text{across-seed SE}.
$$

   *If not satisfied: stop. FGW is not adding signal in this regime, and the
   exploratory leaderboard is at most a hypothesis-generation exercise.*

### Differential predictions (conditional on 1)

2. **The α-sweep curve is non-monotone** — there is an interior α* ∈ (0, 1) where
   the metric peaks. (H_α.) α=0 sets the lower bound (= baseline by construction);
   α=1 isolates the GW contribution.
3. **The primary configuration is at or near the top of the exploratory leaderboard**
   — the most representation-aligned setup wins. (H_bridge, H_image, H_audio.)
   Other cells reaching this region must clear BH q < 0.05.
4. **`geo_cos` improves over `cos_cos`** for the (CLIP, CLAP) pair on rank-sensitive
   metrics. (H_geo.) Triplet agreement should be invariant; recall@k may differ.
5. **Structural metrics correlate with semantic metrics across the grid** —
   evaluated **at fixed α = α***, not pooled across α. Spearman
   `ρ(pearson_dist_corr, recall@1_lex)` over the 72-cell (image × audio × bridge
   × cost × agg) slice at the selected α*, with permutation p-value < 0.05
   (cell labels shuffled 1000× to derive the null).

   *Why fixed α matters.* α directly trades off the GW (structural) and W
   (semantic-via-bridge) objectives, so pooling across α produces a structural
   ↔ semantic anti-correlation that is purely an artifact of the FGW
   formulation. The encoder-quality question — "do good encoders win on both
   metric families?" — is only well-posed at fixed α. For completeness the
   diagnostic also reports the all-α version (which will typically show ρ < 0
   and is informative about the tradeoff steepness, not about alignment
   quality).

   A *positive correlation* with `p < 0.05` at fixed α* is evidence that
   structural diagnostics are informative about the task; a near-zero or
   negative correlation suggests FGW is solving its own objective without
   producing a meaningful matching.
6. **Structural metrics exceed the random-π null.** For the primary configuration,
   `pearson_dist_corr > null_pearson_hi`. This rules out the trivial explanation
   "any permutation has roughly that pearson on isotropic spaces."

### Pre-conditions to even read the headline

- **Dataset overlap diagnostic** (§5, `dataset_overlap.py`) shows the per-image
  distribution of `max_j cos(cap_img_i, cap_aud_j)` under one witness encoder,
  before any FGW. If most images have `max_j ≈ 0`, no algorithm can do well —
  results in that regime should be reported with a "low ambient signal" caveat.
- **Multi-seed runs completed** with ≥5 seeds (§6.3). Single-seed plots show
  bootstrap CIs only and do *not* support H_FGW.

The order matters: (1) is necessary and primary; (2)–(6) are differential
predictions over the exploratory grid, conditional on (1) holding.
