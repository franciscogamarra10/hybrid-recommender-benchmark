# Hybrid Recommender Benchmark: LightFM vs PyTorch

A controlled comparison of feature-augmented matrix factorization (LightFM)
against modern PyTorch hybrid rankers (Sahara YouTubeRanking, Surprise SVD++)
on MovieLens-100k. Same split, same seed, same evaluation protocol. Every
number in this repository was measured; nothing is imported from papers.

The repository answers three questions with numbers instead of intuition:

1. How much ranking quality do metadata features actually buy?
2. Does a PyTorch hybrid ranker match a Cython hybrid ranker at equal
   capacity?
3. What is the runtime cost of migrating?

It also draws a hard line between retrieval (what to show) and causal
decision-making (what to do), because the two are routinely confused in
practice and the confusion is expensive.

## Key findings

| Finding | Evidence |
|---|---|
| Features are worth about **0.14 AUC** | LightFM WARP with features: **0.9331**. Surprise SVD++ IDs-only: **0.7924**. |
| Train-only aggregates **do not help** at matched capacity | Sahara with aggregates: 0.8700. Without: 0.8710. Delta is negative and costs 68 extra seconds. |
| **WARP beats cross-entropy** on top-K ranking | LightFM WARP: 0.906 in 17s. Sahara YouTubeRanking: 0.871 in 776s. |
| **Cython is 45× faster** than PyTorch on CPU | 17s vs 776s at comparable AUC. The gap is structural, not tunable. |
| **Rating prediction is not ranking** | Surprise RMSE 0.344 but its P@10 is not comparable to LightFM's because the candidate pools differ. |

## Final results

| Model | Framework | Features | Test AUC | Train (CPU) |
|---|---|---|---|---|
| **LightFM WARP** | Cython+OpenMP | content | **0.9331** | **18s** |
| Sahara YouTubeRanking | PyTorch | content | 0.8715 | 576s |
| Sahara YouTubeRanking | PyTorch | content + aggregates | 0.8725 | 606s |
| Surprise SVD++ | Python/Cython | IDs only | 0.7840 | 261s |

AUC is the only metric directly comparable across frameworks. P@10 and
R@10 use different candidate pools: LightFM's single-positive protocol,
Sahara's full-negative sampling, Surprise's 1-positive + 100-negatives.
Those metrics are reported inside the notebooks and read intra-framework
only.

# Extensions and Cross-Industry Applications

The pipeline in this repository generalizes to any domain where a decision
maker chooses between alternatives and observes an outcome. The mapping
is straightforward, but the choice of model is not. It depends on whether
the target is a preference or an outcome, and whether the domain is
feature-rich or interaction-rich.

## The core abstraction

| Benchmark | Generic |
|---|---|
| User | Decision maker or context |
| Item | Alternative |
| Rating | Outcome of choosing the alternative |
| User features | Context of the decision |
| Item features | Attributes of the alternative |

## Direct ports

### E-commerce

Target becomes "clicked, added to cart, purchased" rather than
"rated ≥ 3". WARP or BPR are the natural losses. Content features
(category tree, price bucket, brand, image embeddings) fit the same
pipeline.

### Media and streaming

Target is watch time, completion rate, or thumbs up. Title n-grams become
plot embeddings or tag vectors. LightFM handles the tabular version; a
two-tower model with sentence-transformers handles the sequence version.

## Domains with causal targets

These domains are sometimes described as recommendation problems, but the
target is not a preference. It is an outcome produced by the system, and
the action that was observed is one of many that could have been taken.
Three things change.

**Selection is not random.** Completions are chosen by engineers;
treatments are chosen by clinicians. The assignment correlates with the
context, so a model trained on (context, action, outcome) learns a
confounded association, not a causal effect. Recommending the
historically best-performing action for a marginal context will
underperform.

**The score is not a decision.** A LightFM score in this domain answers
"what is the expected outcome under this action, given the historical
selection process." It does not answer "what would the outcome be if we
chose this action now." The first is an association; the second is what a
decision requires.

**The loss is wrong for the task.** WARP and BPR optimize ranking under
a preference assumption. For outcome prediction, an MSE or a survival
loss is appropriate, and for treatment effect estimation, a
doubly-robust or uplift loss is required. AUC and P@10 do not measure
what matters here.

### Oil and gas

| Benchmark | Oil & gas |
|---|---|
| User | Well, field, or operator |
| Item | Completion strategy, treatment, configuration |
| Rating | Normalized production, EUR, NPV |
| User features | Geology (porosity, permeability, lithology) |
| Item features | Operational parameters (depth, fracture design, proppant) |
| Cold-start | New well with no production history |
| Drift | Reservoir depletion, regime change |

### Healthcare

| Benchmark | Healthcare |
|---|---|
| User | Patient |
| Item | Treatment protocol |
| Rating | Outcome (recovery, side effects) |
| User features | Diagnosis, biomarkers, age, comorbidities |
| Item features | Drug class, dosage, contraindications |

In both domains the correct estimator for the decision layer is uplift
modeling or causal bandits (`causalml`, `econml`), not LightFM. LightFM
models preferences. There is no preference in a well or a patient: there
is production, and there is recovery. The feature pipeline from this
repository is reusable verbatim; the estimator should be swapped for one
that estimates treatment effects.

### Finance

Product recommendation (cards, loans, investment products) is a direct
port with one twist: the target must be a customer preference (click,
application) and not a business outcome (default, fraud). The latter is
a causal problem, not a ranking problem. Keep those two separate in the
data model.

## What transfers, what does not

| Component | Retrieval layer | Decision layer |
|---|---|---|
| Feature engineering pipeline | Yes, unchanged | Yes, unchanged |
| Matrix factorization backbone | Yes | Not applicable |
| Cold-start from features | Yes | Depends on identification |
| Train-only aggregates | With caution: they encode the assignment policy | Usually a confounder, must be handled explicitly |
| WARP / BPR / cross-entropy losses | Yes | No; use outcome or effect losses |
| AUC, P@10, R@10 | Yes | No; use Qini, AUUC, PEHE |
| Train/serve bundle | Yes | Yes, plus identification assumptions in the model card |

## The general rule

**If the decision is what to show, use a recommender. If the decision is
what to do, use causal modeling. If the decision is what to try among
many options, use both, stacked.**

The benchmark in this repository answers the first case with concrete
numbers. The second case is a different problem with its own literature.
The third is the interesting one in practice, and it is the reason the
retrieval layer should be built to hand off cleanly to whatever comes
next.

