# FCG Algorithm - Visual Flowchart

## High-Level Algorithm Flow

```
                    ┌─────────────────────┐
                    │   Training Data     │
                    │    (e.g., 14K)      │
                    └──────────┬──────────┘
                               │
                ┌──────────────▼──────────────┐
                │   STEP 1: CLUSTERING        │
                │   (Reduce & Diversify)      │
                └──────────────┬──────────────┘
                               │
        ┌──────────────┬───────┴────────┬──────────────┐
        │              │                │              │
    ┌───▼──┐      ┌───▼──┐       ┌────▼──┐       ┌───▼──┐
    │F,Neg │      │F,Pos │       │M,Neg  │       │M,Pos │
    │3000+ │      │2000+ │       │5000+  │       │4000+ │
    │  │   │      │  │   │       │  │    │       │  │   │
    │  ▼   │      │  ▼   │       │  ▼    │       │  ▼   │
    │ 40   │      │ 40   │       │  40   │       │  40  │
    └──┬───┘      └──┬───┘       └───┬───┘       └──┬───┘
       │              │               │              │
       └──────────────┴───────┬───────┴──────────────┘
                              │
                ┌─────────────▼──────────────┐
                │   STEP 2: GENETIC          │
                │   EVOLUTION (10 iters)     │
                │   Score-based Selection    │
                └─────────────┬──────────────┘
                              │
              Iteration 1-10:  │
              ┌────────────────▼─────────────────┐
              │ 1. Roulette wheel selection      │
              │    (K=8 samples)                 │
              ├──────────────────────────────────┤
              │ 2. Evaluate on validation set    │
              │    (100 samples)                 │
              ├──────────────────────────────────┤
              │ 3. Calculate:                    │
              │    - Accuracy                    │
              │    - Fairness (Rdp, Reo)        │
              │    - EvolScore = α·ΔPred +     │
              │                 (1-α)·ΔFair    │
              ├──────────────────────────────────┤
              │ 4. Update scores of selected    │
              │    samples (average)            │
              └────────────────┬────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Scores Updated for   │
                    │ Each Subgroup:       │
                    │ F,Neg: [0.5, ...]    │
                    │ F,Pos: [0.3, ...]    │
                    │ M,Neg: [0.4, ...]    │
                    │ M,Pos: [0.6, ...]    │
                    └──────────┬───────────┘
                               │
                ┌──────────────▼──────────────┐
                │   STEP 3: SELECTION         │
                │   (Apply Strategy)          │
                └──────────────┬──────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
    ┌───▼────┐          ┌──────▼──────┐          ┌───▼────┐
    │ STRATEGY S1       │ STRATEGY S2  │          │STRATEGY S3
    │ (Balanced)        │(Minority+Bal)│          │(Minority+Unbal)
    ├─────────┤         ├──────────────┤          ├────────┤
    │Top K/4: │         │Top K/2 from: │          │Top K:  │
    │ -F,Neg  │         │ -F,Neg       │          │ -F,Pos │
    │ -F,Pos  │         │ -F,Pos       │          │        │
    │ -M,Neg  │         │ (BEST!)✨    │          │(BEST✨ │
    │ -M,Pos  │         │              │          │for max │
    └─────────┘         └──────────────┘          │fairness)
                                                   └────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                        ┌──────▼──────┐
                        │  K=8 Final  │
                        │Demonstrations
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │ Create      │
                        │ Prompt with │
                        │ K selected  │
                        │ examples    │
                        └──────┬──────┘
                               │
                        ┌──────▼──────────────────┐
                        │ Evaluate on Test Set    │
                        │ - Accuracy              │
                        │ - Fairness metrics      │
                        └─────────────────────────┘
```

## Detailed Step 1: Clustering Process

```
Input: All Female samples in training set (e.g., 5000 samples)

Step 1a: Feature Extraction
    Sample 1: [Age=25, Workclass=5, Education=8, ...] → [25, 5, 8]
    Sample 2: [Age=45, Workclass=2, Education=10, ...] → [45, 2, 10]
    Sample 3: [Age=35, Workclass=5, Education=9, ...] → [35, 5, 9]
    ...
    Sample 5000: [Age=55, Workclass=7, Education=6, ...] → [55, 7, 6]

Step 1b: K-means Clustering (n=8 clusters)
    
    Feature Space (simplified 2D):
    
    ┌─────────────────────────────────────┐
    │            Age                      │
    │  ▲                                  │
    │  │      C₁●              ●C₂        │
    │  │      /  \            /  \        │
    │  │    ○    ○         ○    ○         │
    │  │   /      \       /      \        │
    │  │  ○        ○     ○        ○       │
    │  │   \      /       \      /        │
    │  │    ○    ○         ○    ○         │
    │  │      \  /            \  /        │
    │  │      C₃●              ●C₄        │
    │  │                                  │
    │  └─────────────────────────────────►
    │           Workclass
    │
    │  (8 cluster centers, with many ○ points)
    └─────────────────────────────────────┘

Step 1c: Select m=5 nearest neighbors per cluster
    
    For Cluster 1 center (25, 5):
        - Find 5 closest samples to center
        - Indices: [42, 157, 234, 289, 401]
    
    For Cluster 2 center (35, 8):
        - Find 5 closest samples to center
        - Indices: [103, 215, 367, 489, 512]
    
    ... (repeat for all 8 clusters)

Step 1d: Combine selections
    Selected indices: [42, 157, 234, 289, 401,     ← From C₁
                      103, 215, 367, 489, 512,     ← From C₂
                      ...
                      ]
    Total selected: 8 clusters × 5 neighbors = 40 samples

Output: From 5000+ samples → 40 representative samples
        (maintaining diversity across age, workclass, education, etc.)
```

## Detailed Step 2: Genetic Evolution Process

```
Initial State:
    Candidates per subgroup: 40 (from clustering)
    Scores per candidate: 0.05 (initial score)
    
    F,Neg: [0.05, 0.05, 0.05, ..., 0.05]  (40 values)
    F,Pos: [0.05, 0.05, 0.05, ..., 0.05]  (40 values)
    M,Neg: [0.05, 0.05, 0.05, ..., 0.05]  (40 values)
    M,Pos: [0.05, 0.05, 0.05, ..., 0.05]  (40 values)

Iteration 1:
    ┌─────────────────────────────────────────────────────┐
    │ 1. Roulette Wheel Selection from F,Neg subgroup     │
    │                                                     │
    │    Scores: [0.05, 0.05, 0.05, 0.05, ..., 0.05]    │
    │    Probabilities: [0.025, 0.025, 0.025, ...]       │
    │                   (equal for all)                   │
    │                                                     │
    │    Randomly select K=8 samples:                     │
    │    → Indices: [2, 7, 12, 19, 23, 31, 35, 38]      │
    ├─────────────────────────────────────────────────────┤
    │ 2. Create Prompt from selected demonstrations       │
    │                                                     │
    │    Prompt:                                          │
    │    "Age: 35, Workclass: Private, ..., Income: ..." │
    │    "Age: 42, Workclass: Gov-fed, ..., Income: ..."│
    │    (K=8 examples total)                             │
    ├─────────────────────────────────────────────────────┤
    │ 3. Evaluate on 100 validation samples              │
    │                                                     │
    │    For sample i in val_data:                        │
    │      - Get LLM prediction                           │
    │      - Compare with ground truth                    │
    │      - Check if prediction is correct               │
    │                                                     │
    │    Results: 75 correct / 100 = 75% accuracy        │
    ├─────────────────────────────────────────────────────┤
    │ 4. Calculate Fairness Metrics                       │
    │                                                     │
    │    Separate predictions by gender:                  │
    │    Female accuracy: 78% (DP₀)                      │
    │    Male accuracy: 72% (DP₁)                        │
    │    Δdp = |0.78 - 0.72| = 0.06 (lower is fair)     │
    │    Rdp = min(0.78, 0.72)/max(...) = 0.923 ↑       │
    │                                                     │
    │    Fairness score: Rdp = 0.923                     │
    ├─────────────────────────────────────────────────────┤
    │ 5. Calculate EvolScore                              │
    │                                                     │
    │    ΔPred = max(0.75 - 0.50, 0.05) = 0.25           │
    │    ΔFair = max(0.923 - 0.20, 0.05) = 0.723        │
    │    EvolScore = 0.5 × 0.25 + 0.5 × 0.723 = 0.487   │
    ├─────────────────────────────────────────────────────┤
    │ 6. Update Scores of Selected Samples                │
    │                                                     │
    │    For each selected index (2, 7, 12, 19, ...):    │
    │      new_score = (old_score + evo_score) / 2       │
    │      = (0.05 + 0.487) / 2 = 0.269                  │
    │                                                     │
    │    F,Neg after iteration 1:                         │
    │    [0.05, 0.05, 0.269, 0.05, ..., 0.269, ...]    │
    │           ↑                ↑
    │         (unchanged)    (updated: was selected)     │
    └─────────────────────────────────────────────────────┘

Iteration 2:
    ┌─────────────────────────────────────────────────────┐
    │ Updated Scores: [0.05, 0.05, 0.269, 0.05, ...]    │
    │ Updated Probabilities (normalized):                 │
    │   [0.020, 0.020, 0.108, 0.020, ..., 0.108, ...]   │
    │                           ↑                ↑
    │                    (higher prob)     (higher prob)  │
    │                                                     │
    │ Roulette Wheel: samples with score=0.269 now       │
    │                have ~5x higher selection prob!      │
    │                                                     │
    │ → Selection now biased toward "good" samples       │
    │   (more likely to select indices 2 and 23 again)   │
    └─────────────────────────────────────────────────────┘

Iteration 3-10: Repeat...
    
    Result: After 10 iterations:
    ┌─────────────────────────────────────┐
    │ Final Scores (example values):       │
    │                                     │
    │ F,Neg: [0.05, 0.08, 0.67, 0.15,    │
    │         0.12, 0.55, 0.06, 0.72,    │
    │         ...]                        │
    │ F,Pos: [0.09, 0.68, 0.04, 0.61,    │
    │         ...]                        │
    │ M,Neg: [0.10, 0.52, 0.07, 0.45,    │
    │         ...]                        │
    │ M,Pos: [0.63, 0.08, 0.71, 0.06,    │
    │         ...]                        │
    │                                     │
    │ → High scores (0.6-0.7): excellent │
    │   demonstrations for fairness       │
    │ → Low scores (0.05-0.10): poor      │
    │   demonstrations                    │
    └─────────────────────────────────────┘
```

## Detailed Step 3: Selection by Strategy

```
Available candidates with scores:
    F,Neg: [0.05, 0.08, 0.67✓, 0.15, 0.12, 0.55✓, 0.06, 0.72✓, ...]
    F,Pos: [0.09, 0.68✓, 0.04, 0.61✓, ...]
    M,Neg: [0.10, 0.52✓, 0.07, 0.45✓, ...]
    M,Pos: [0.63✓, 0.08, 0.71✓, 0.06, ...]

STRATEGY S1: Balanced Selection (rz=0.5, ry=0.5)
    ┌────────────────────────────────────────┐
    │ Select Top K/4 from each of 4 groups  │
    │                                        │
    │ Top 2 from F,Neg: [0.72, 0.67]        │
    │ Top 2 from F,Pos: [0.68, 0.61]        │
    │ Top 2 from M,Neg: [0.52, 0.45]        │
    │ Top 2 from M,Pos: [0.71, 0.63]        │
    │                                        │
    │ Total K=8 demonstrations               │
    │ Composition: 50% Female, 50% Male     │
    │            50% Negative, 50% Positive│
    └────────────────────────────────────────┘
    
    Result: Balanced but mediocre fairness

STRATEGY S2: Minority Prioritization (rz=1, ry=0.5) ✨ BEST
    ┌────────────────────────────────────────┐
    │ Select Top K/2 from FEMALE subgroups   │
    │                                        │
    │ Top 4 from F,Neg: [0.72, 0.67, 0.55, ..]
    │ Top 4 from F,Pos: [0.68, 0.61, 0.58, ..]
    │                                        │
    │ Total K=8 demonstrations               │
    │ Composition: 100% Female               │
    │            50% Negative, 50% Positive│
    │                                        │
    │ Why this works:                       │
    │ - LLM "sees" minority group often     │
    │ - Learns to treat them fairly         │
    │ - Fairness improves massively!        │
    │                                        │
    │ Paper result: Δdp improves 70%!       │
    └────────────────────────────────────────┘
    
    Result: Dramatically improved fairness!

STRATEGY S3: Maximum Minority (rz=1, ry=1)
    ┌────────────────────────────────────────┐
    │ Select Top K from FEMALE POSITIVE only │
    │                                        │
    │ Top 8 from F,Pos:                      │
    │   [0.68, 0.61, 0.58, 0.50, 0.48, ...]│
    │                                        │
    │ Total K=8 demonstrations               │
    │ Composition: 100% Female, 100% Positive
    │                                        │
    │ Maximum fairness but most imbalanced   │
    │ predictions (LLM biased to predict    │
    │ positive)                              │
    └────────────────────────────────────────┘
    
    Result: Best fairness, trade-off on accuracy
```

## Real Example: From Paper

```
Adult Income Dataset Results:

WITHOUT FCG (Random Selection):
    │
    │  Fairness Rdp: 0.677
    │  ├─ Female positive predictions: 85%
    │  └─ Male positive predictions: 72%
    │     (13% difference → unfair)

WITH FCG + STRATEGY S2:
    │
    │  Fairness Rdp: 0.894  ← 32% improvement!
    │  ├─ Female positive predictions: 89%
    │  └─ Male positive predictions: 88%
    │     (1% difference → much fairer!)
    │
    │  Accuracy: 0.779 (0.733 → 0.779 with FCG clustering)
    │            (slight improvement!)

Trade-off Summary:
    ✓ Fairness: +32%
    ✓ Accuracy: +6%
    = WIN-WIN situation!
```
