
# When does spatial context matter? GNNs vs MLPs on Visium spatial transcriptomics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USERNAME>/spatial-gnn-visium/blob/main/spatial_gnn.ipynb)

**Question:** Does knowing a spot's physical neighbors help predict which brain region it belongs to, beyond what its own gene expression already says?

**Answer:** Not when expression is rich — but as the expression signal weakens, a graph neural network that uses neighbors outperforms an expression-only model by up to 24 accuracy points. Spatial context acts as a denoiser: it helps most when each spot carries partial information.

![Spatial context compensates for weak expression signal](figures/spatial_vs_expression.png)

## Data

10x Visium mouse brain section (H&E), loaded via `squidpy.datasets.visium_hne_adata()`.

- 2,688 spots, each with an expression profile and (x, y) tissue coordinates
- 15 annotated regions (cortex layers, hippocampus, thalamus, fiber tract, etc.)
- Features: top highly variable genes (log-normalized)

## Approach

Node classification: predict each spot's region label. All models share the same training loop, splits, and hyperparameters — the only difference is whether, and how, a spot sees its neighbors.

| Model | What each spot sees |
|---|---|
| **MLP** | Its own expression only |
| **GCN** (`GCNConv`) | The average of itself and its neighbors |
| **GraphSAGE** (`SAGEConv`) | Its own features and the neighbor mean, through separate learned weights |

**Graph:** each spot connected to its 6 hexagonal neighbors (`squidpy.gr.spatial_neighbors`, grid mode). Edge homophily = 0.76 (76% of edges join same-region spots).

**Training:** 2-layer models, hidden size 128, dropout 0.5, Adam (lr 1e-3, weight decay 5e-4), full-batch, transductive setting. 80/20 stratified split; results averaged over 10 random splits with paired comparisons.

## Results

### 1. GCN underperforms; GraphSAGE recovers

With 2,000 genes on a single split, GCN scored **89.4%** vs the MLP's **92.8%**, and failed to fit the training set (95% train accuracy). GCN replaces each spot's features with a neighborhood average, so a spot's own signal becomes ~1/7 of its input, and boundaries between regions blur (oversmoothing). GraphSAGE keeps self and neighbor information on separate tracks and recovered full training fit.

### 2. With full expression, spatial context adds nothing

Across 10 splits at 2,000 genes: **MLP 0.943 ± 0.013**, **GraphSAGE 0.939 ± 0.012** (SAGE wins 2/10). Expected: the region labels were derived from expression, so a spot's own profile already determines its label.

### 3. As expression weakens, spatial context takes over

Features were restricted to the top *k* most variable genes:

| Genes | MLP | GraphSAGE | Gap | SAGE wins (of 10) |
|---:|---:|---:|---:|---:|
| 2000 | 0.942 | 0.938 | −0.003 | 3 |
| 500 | 0.917 | 0.926 | +0.009 | 8 |
| 100 | 0.774 | 0.875 | +0.101 | 10 |
| 50 | 0.525 | 0.764 | **+0.239** | 10 |
| 20 | 0.346 | 0.546 | +0.200 | 10 |
| 10 | 0.246 | 0.354 | +0.108 | 10 |

The gap follows an inverted U. Neighbor aggregation fills in missing information when each spot has *partial* signal, but cannot create signal that isn't there — at 10 genes every neighbor is weak too.

## Takeaways

- A layer's built-in assumption about the data matters as much as model capacity: GCN assumes a spot resembles its neighborhood average; GraphSAGE learns how much to trust it.
- Spatial information is most valuable in the sparse-signal regime — the setting of high-resolution platforms (e.g., Visium HD, Xenium) where each spot or cell captures fewer transcripts.

## Limitations

- Region labels were derived from expression clustering, which caps how much location can help at full gene coverage. Anatomically annotated labels would be a stronger test.
- Gene restriction is a proxy for sparse signal; downsampling counts would more directly mimic low-depth data.
- Single tissue section.

## Run it

Open the notebook in Colab (badge above), set **Runtime → T4 GPU**, and run all cells. Dependencies: `torch`, `torch_geometric`, `squidpy`, `scanpy`.
