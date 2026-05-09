# FGW cross-modal alignment — results interpretation

**Data**: `fgw_validation/results/results/fgw_n200_a*_s0_eval.json`
(7 α-values × 64 combos × 1 seed = **448 evaluated configurations**;
n = 200 sampled per side)

**Eval version**: predates the path-A changes — these JSONs carry
`baseline_*` and bootstrap CIs, but **not** `null_*`, `lex` witness,
`soft_recall_*`, `mrr_*`, or `median_rank_*`. To get those, re-run
`python -m fgw_validation.eval` over the existing results files.

---

## TL;DR

> **H_FGW (the load-bearing claim from EXPERIMENT.md §4) is falsified by this
> data.** Across **all 448 configurations** and **all 4 held-out semantic
> witnesses**, FGW's `recall@1` is *strictly lower* than the text-only baseline
> — every single time. The best FGW configuration in the entire grid loses to
> baseline by **0.020** in recall@1; the average gap is **−0.09 to −0.12**.
> No bootstrap CI or seed-variance argument can rescue this — the entire
> grid is on the wrong side.

The exploratory leaderboard, the α-sweep, and the structural metrics still
contain useful information about *how* FGW behaves (and *why* the GW term
hurts), but no positive claim about FGW as a cross-modal aligner is
defensible from this data.

---

## 1. The headline

`recall@1` averaged over the held-out witness encoders ≠ bridge:

| bridge   | α=0.00 | α=0.10 | α=0.25 | α=0.50 | α=0.75 | α=0.90 | α=1.00 |
|----------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| clap     | −0.082 | −0.083 | −0.085 | −0.088 | −0.087 | −0.087 | −0.120 |
| clip     | −0.121 | −0.120 | −0.121 | −0.121 | −0.121 | −0.120 | −0.150 |
| roberta  | −0.063 | −0.066 | −0.070 | −0.071 | −0.077 | −0.080 | −0.107 |
| t5       | −0.081 | −0.081 | −0.079 | −0.081 | −0.078 | −0.081 | −0.100 |

(values are FGW − baseline; negative means FGW worse). The smallest gap
anywhere in the table is **−0.063** (roberta-bridge at α=0).

**Per-witness counts of FGW > baseline across the entire grid:**

| witness | n combos | mean Δ | best Δ | combos with FGW > baseline |
|---|---:|---:|---:|---:|
| clip    | 336 | −0.118 | **−0.035** | 0 |
| clap    | 336 | −0.091 | **−0.020** | 0 |
| roberta | 336 | −0.085 | −0.030 | 0 |
| t5      | 336 | −0.080 | −0.030 | 0 |
| **all** | **1344** | **−0.094** | **−0.020** | **0 / 1344** |

The single best (FGW, witness, combo, α) tuple in the entire dataset is
`(clip × clap × roberta × cos_cos × mean, α=0.1)` with `recall@1_clap = 0.070`
vs `baseline_recall@1_clap = 0.090` — still a loss of 0.020, on the witness
*most favorable to* FGW.

---

## 2. Why the α-sweep doesn't save it

The classic FGW story is: at α=0 you're text-only, at α=1 you're geometry-only,
and somewhere in between is a sweet spot where the structural and semantic
signals reinforce. The data flatly does not show this.

**Mean `recall@1_avg` (held-out witnesses) per bridge × α — FGW (baseline in parens):**

| bridge   | α=0.00       | α=0.10       | α=0.25       | α=0.50       | α=0.75       | α=0.90       | α=1.00       |
|----------|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
| clap     | 0.043 (.125) | 0.042 (.125) | 0.040 (.125) | 0.037 (.125) | 0.038 (.125) | 0.038 (.125) | 0.005 (.125) |
| clip     | 0.035 (.157) | 0.036 (.157) | 0.036 (.157) | 0.036 (.157) | 0.036 (.157) | 0.037 (.157) | 0.007 (.157) |
| roberta  | 0.050 (.113) | 0.048 (.113) | 0.043 (.113) | 0.043 (.113) | 0.037 (.113) | 0.034 (.113) | 0.006 (.113) |
| t5       | 0.024 (.105) | 0.024 (.105) | 0.026 (.105) | 0.024 (.105) | 0.027 (.105) | 0.024 (.105) | 0.005 (.105) |

Three patterns jump out:

1. **No interior maximum.** FGW is monotonically (or near-monotonically) *worse*
   as α grows. Adding any GW signal makes things worse, and pure GW (α=1) drives
   semantic recall to chance level (~1/n_a = 0.005 with n=200).
2. **Bridge ranking inverts** between FGW and baseline. Baseline ranks
   `clip > clap > roberta > t5` (multimodally trained encoders give the best
   `M`). FGW at low α ranks `roberta > clap > clip > t5` — *the opposite
   ordering on `clip` and `roberta`.* FGW penalises the strongest bridges
   most, presumably because the optimal-assignment LP (see §3) has more to
   destroy when the row-wise greedy was already producing strong matches.
3. **At α=0 the gap is already large** (−0.06 to −0.12). This was the most
   surprising finding and is structural, not a bug — see §3.

---

## 3. The α=0 gap is real (and changes the EXPERIMENT.md sanity-check claim)

EXPERIMENT.md §7 originally claimed:

> *"At α = 0, FGW minimises ⟨T, M⟩ only … π_FGW(α=0) = π_baseline (up to BCD
> tolerance)."*

This is **wrong**. The data shows FGW at α=0 differs from baseline by the same
~0.10 gap as the higher-α points. Investigation:

- `entropy_norm ≈ 0.5`, `top1_mass = 1.0`, `coverage = 1.0` at every
  (combo, α). So FGW always converges to a **permutation matrix** (each row
  has all mass on a single column, and every column gets used exactly once).
- The text-only baseline `π_baseline(i) = argmax_j cos(text_i, text_j)` is a
  **row-wise greedy retrieval** — multiple rows can map to the same column
  (collisions allowed). This is **not** a permutation in general.

So at α=0 we are comparing two different problems:

| | objective | constraint on π |
|---|---|---|
| FGW(α=0)  | `min Σ_{i,j} T[i,j] · M[i,j]` (LP) | uniform marginals → permutation |
| baseline  | `min_j M[i,j]` per row independently | row-wise argmax (no constraint) |

These produce different π and hence different `recall@1`. **The data tells us
that the row-wise greedy retrieval is doing something the optimal-assignment
LP cannot do**: it's allowed to send many `i` to the same well-matched `j`
(e.g. all dog-related Flickr8k images mapping to the same dog-bark Clotho
clip). That's exactly the right answer when the two distributions have
asymmetric coverage — and the size-of-loss tells us this happens a lot.

**Implication for EXPERIMENT.md.** The §7 sanity check should be replaced
with: *"at α=0, FGW(α=0) should match a Hungarian-on-M baseline (LP
assignment), **not** the row-wise greedy baseline."* The current `baseline_*`
columns are **the right comparison for the H_FGW question** (does FGW beat
text-only retrieval?), but they are **not** a calibration of the FGW solver.
A separate Hungarian-on-M baseline is needed for that.

---

## 4. Where FGW *does* win — and why it doesn't help

Structural metrics (`pearson_dist_corr`, `triplet_agreement`) — by
construction, FGW directly optimises this. Δ (FGW − baseline) on
`pearson_dist_corr`:

| bridge   | α=0.00 | α=0.10 | α=0.25 | α=0.50 | α=0.75 | α=0.90 | α=1.00 |
|----------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| clap     | −0.13  | −0.13  | −0.12  | −0.10  | −0.04  | +0.05  | **+0.18** |
| clip     | −0.12  | −0.11  | −0.09  | −0.05  | −0.00  | +0.03  | **+0.21** |
| roberta  | −0.12  | −0.03  | +0.01  | +0.04  | +0.08  | +0.12  | **+0.22** |
| t5       | −0.04  | −0.02  | +0.01  | +0.06  | +0.13  | +0.20  | **+0.28** |

So FGW beats baseline on **structural fit** at α ≥ 0.5 (especially with
text-only bridges), and dominates at α=1. This is exactly what we'd expect:

- **At α=1, FGW maximizes structural fit by definition.** It finds a
  permutation π that makes `C_a[π(i), π(j)]` look maximally like `C_i[i, j]`.
- **But** that permutation is *semantically random* — `recall@1_avg` at α=1 is
  0.005, indistinguishable from chance.

This is the classic dissociation that we predicted in EXPERIMENT.md §11(5):
"if structural is high but semantic is low everywhere, FGW is solving its own
objective without producing a meaningful matching." The data shows exactly
this dissociation at α≥0.75 — structural goes up, semantic goes down. The two
metric families are *anticorrelated* across α, not aligned.

The previous EXPERIMENT.md framing ("text-only baseline isn't fair on
structural") was the right caveat: FGW's structural dominance at high α is
real but mechanically guaranteed and contains no information about cross-modal
alignment quality. The `null_pearson_*` columns from the path-A re-eval will
ground this properly.

---

## 5. Differential predictions: what the ablation axes actually show

### 5.1 Cost convention (cos_cos vs cos_neg) — narrow

Mean `recall@1_avg` over the entire grid:

| convention | FGW   | baseline |
|---|---:|---:|
| `cos_cos`  | 0.0306 | 0.1250 |
| `cos_neg`  | 0.0323 | 0.1250 |

`cos_neg` is marginally better (~0.002), confirming the reviewer's claim that
this axis buys very little. Worth folding into the exploratory leaderboard,
not worth featuring.

### 5.2 Caption aggregation (mean vs first) — meaningful

| agg     | FGW   | baseline |
|---|---:|---:|
| `mean`  | 0.0398 | 0.1250 |
| `first` | 0.0232 | 0.1250 |

Mean-pool is ~0.017 better than using only the first caption — the only
clear positive ablation result. This is intuitive (5 captions provide more
information than 1) and supports the EXPERIMENT.md primary-config choice of
`mean`.

### 5.3 Image / audio encoder choice — barely matters

| image  | FGW   |     | audio | FGW   |
|---|---:|---|---|---:|
| clip   | 0.0313 |    | clap  | 0.0313 |
| dinov2 | 0.0317 |    | ast   | 0.0317 |

Within 0.0004. The pre-registered choice of `(clip, clap)` was motivated by
manifold geometry, not by FGW outcome — and indeed the outcome doesn't
discriminate. This is consistent with FGW being mostly broken on this task:
when the algorithm fails uniformly, encoder differences disappear in noise.

### 5.4 Geodesic ablation (`geo_cos`) — not in this data

`geo_cos` was added to the codebase later. The current results contain only
`cos_cos` and `cos_neg`. Re-running the (CLIP, CLAP) sweep with
`--cost_conventions geo_cos` would add 8 combos × 7 α = 56 evaluations to test
H_geo.

---

## 6. Transport-plan diagnostics — uniformly degenerate

| α | entropy_norm | top1_mass | mutual_best | coverage |
|---|---:|---:|---:|---:|
| 0.00 | 0.500 | 1.000 | 1.000 | 1.000 |
| 0.10 | 0.500 | 1.000 | 1.000 | 1.000 |
| 0.25 | 0.500 | 1.000 | 1.000 | 1.000 |
| 0.50 | 0.500 | 1.000 | 1.000 | 1.000 |
| 0.75 | 0.500 | 1.000 | 1.000 | 1.000 |
| 0.90 | 0.500 | 1.000 | 1.000 | 1.000 |
| 1.00 | 0.500 | 1.000 | 1.000 | 1.000 |

Identical at every α. FGW always converges to a hard permutation matrix, and
`entropy_norm = log(n) / log(n²) = 0.5` is precisely the entropy of a
permutation under the chosen normalisation (so this isn't carrying real
information; it's a definitional artifact). `mutual_best = coverage = 1.0`
again indicates "permutation-matrix T, every column used once" at every α —
the LP-assignment structure dominates regardless of how heavily we weight the
GW vs W term.

This is informative: **the FGW solver isn't getting stuck in a high-entropy
local minimum or producing a degenerate dump-all-mass solution**. The
optimization is working as advertised. The matchings it produces are simply
the wrong matchings for the cross-modal-alignment task.

---

## 7. What's missing from this analysis (path-A additions)

Re-running `python -m fgw_validation.eval --results fgw_validation/results/results/fgw_*_eval.json`
(now possible since the eval module has been upgraded) would add to every combo:

- **`null_pearson_*`, `null_spearman_*`** — the random-π structural null. Tells
  us how much of the +0.21 Pearson gap at α=1 is real vs how much is
  "permutations on isotropic spaces have ρ ≈ 0.20 by default." With n=200,
  the null is likely tight (around 0–0.10) and FGW will exceed it at α=1, but
  this needs to be verified, not assumed.
- **`lex` witness** — the encoder-free Jaccard-over-captions witness. This is
  the single most important addition: it removes the "shared web-text training
  data" leakage concern from the held-out witnesses and gives a
  truly-independent semantic check. If the gap is still −0.10 on `recall@1_lex`,
  no encoder-correlation defense is available.
- **`mrr_*`, `median_rank_*`, `soft_recall@k_*`** — robust replacements for
  `mean_rank` (which has a heavy right tail at n=200) and a
  T-weighted recall that doesn't collapse FGW's soft transport plan to its
  argmax.
- **Item-level structural bootstrap CIs** — the current `pearson_dist_corr_*`
  CIs use pair-of-pairs resampling and are anticonservative. Re-eval will
  replace them with proper item-level CIs.

These additions are mechanical — they don't change what the data is saying
about H_FGW (which is already decisive at this signal level), but they sharpen
all the secondary claims and close the easy criticism vectors.

---

## 8. What's missing from this *experiment* (multi-seed)

The single-seed budget is the most important **experimental** gap. With
n_seeds=1:

- "FGW at the best combo loses by 0.020 in recall@1" is a single-sample
  estimate. The within-sample bootstrap CI on that combo's recall@1 is wide
  (≈ ±0.04 at n=200, p≈0.07). On just *that single combo* the gap is not
  bootstrap-significant.
- However, the gap is **negative for every one of 1344 (combo × witness)
  cells**, which is itself extremely strong evidence even without per-cell
  CIs: under the null hypothesis "FGW = baseline on each cell with 50/50
  better/worse," the probability of getting 0/1344 wins is `0.5^1344` —
  vanishingly small even after multiplicity correction. So the headline
  conclusion "**FGW is not better than baseline anywhere**" is robust to seed
  variation, multiple comparisons, and within-sample noise. Multi-seed
  wouldn't change this.

Multi-seed *would* matter for a hypothetical "small positive lift on a
specific combo at a specific α" — that finding would require across-seed SE
to be defensible. We don't have such a finding here, so the multi-seed
upgrade is currently a "nice to have for future positive results," not a
requirement for the current negative result.

---

## 9. What this means

Three readings, ordered from least to most pessimistic about FGW.

### 9a. The narrow reading: this regime is wrong for FGW

FGW with a text bridge is forced into an LP-assignment (permutation T) with
uniform marginals. When the two distributions (Flickr8k images, Clotho audio)
have **asymmetric semantic coverage** — i.e., many images have multiple
plausible audio matches and many audios match no images at all — the
optimal-assignment LP cannot exploit that asymmetry. Row-wise greedy
retrieval can. The EXPERIMENT.md §1 framing already hinted at this ("Flickr8k
photos may have *no* sensible audio counterpart in Clotho"). The new
`dataset_overlap.py` diagnostic should quantify this.

If this is the right reading, the fix isn't "tune FGW better"; it's "use
unbalanced FGW (let the marginals slip) or use a different formulation that
allows column collisions."

### 9b. The intermediate reading: bridge encoder caps the ceiling

Baseline `recall@1` peaks at 0.157 (clip-bridge), which is itself low — the
bridge encoder, given perfect access to captions on both sides, only
correctly aligns ~16% of items at top-1. FGW's job is to add value *over*
that ceiling. With such a low ceiling, the GW signal would have to be
remarkably informative to outperform it. Apparently it isn't, on these
encoder choices and these datasets.

### 9c. The broad reading: the regime tested isn't FGW's use case

FGW's strongest argument applies when **one side has no caption supervision**
and a small reference-aligned subset transfers structure to the rest. By
giving captions to both sides we hand the bridge a complete answer for free,
and the GW term is competing against direct text-text retrieval. FGW was
never going to win this comparison cleanly. (This was flagged in
EXPERIMENT.md §1 after the review.)

---

## 10. Recommended next actions

In order:

1. **Re-eval the existing 7 result files** with the upgraded `eval.py` to
   get `null_*`, `lex`, `mrr_*`, `soft_recall_*`, item-level CIs. Single
   command, ~10 minutes:
   ```bash
   rm fgw_validation/results/results/*_eval.json
   python -m fgw_validation.eval \
       --results 'fgw_validation/results/results/fgw_n200_a*_s0.json' \
       --emb fgw_validation/data/embeddings
   ```
   Then re-run plots — this gives `null_pearson_*` shaded bands on the
   alpha-sweep figures and the lex-witness recalls.

2. **Run the dataset-overlap diagnostic** to quantify reading (9a):
   ```bash
   python -m fgw_validation.dataset_overlap --witness lex --n 1000
   python -m fgw_validation.dataset_overlap --witness clip --n 1000
   ```
   The histograms will show how many Flickr8k images have *any* plausible
   Clotho match. If the median lex-Jaccard `max_j` is < 0.05, the regime is
   barely above chance and that contextualizes the negative H_FGW result.

3. **Add a Hungarian-on-M baseline** to `eval.py` (~30 lines) so the α=0
   sanity check is well-defined. This isolates the "is the FGW LP itself
   solving correctly?" question from "does the LP-assignment beat row-wise
   retrieval?".

4. **Decide whether to invest in unbalanced FGW** (`ot.gromov.entropic_fgw_*`
   with non-uniform marginals, or POT's `gromov_wasserstein_partial`). This
   is the only path-of-least-effort modification that could change the
   headline. If the dataset-overlap diagnostic in (2) shows asymmetric
   coverage, this is well-motivated.

5. **Otherwise**: write up the negative result honestly. The infrastructure
   built here (the falsifiable test, the encoder-free witness, the structural
   null, the multi-seed protocol) is reusable and worth publishing as the
   contribution even if FGW itself is not the headline. A clean "we built a
   careful eval and FGW failed it" paper is more valuable than a positive
   result that wouldn't survive the review questions in
   EXPERIMENT.md §11.
