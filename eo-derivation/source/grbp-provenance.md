# Derivation for GrBP: Appendix I source map

Source: `BaiLiping/EO_Writing_Long_Version`, commit
`d0a380a9caf81a36b88886efa91b5afd9bb7bdb5`, `sections/sec_appendixI.tex`.
The 129-line excerpt in `appendix-I-original.tex` matches that revision byte for byte.

Full appendix file SHA-256: `9bf37e14bf9be93e4e4d9227a1520ccad46c5517793a8477f4040e93b43ce0d4`.
Excerpt SHA-256: `61201bdb496727ab5eca877e6b5814c66b1ce7f1f7649ec626419871e29265cb`.

The page adds `#grbp` after S4. The 50-slide companion adds 17 slides after the
original S4 factor graph, beginning at `#grbp-preclustering`. The original
S1–S4 derivation sections and teaching slides are preserved.

| Source item | Original lines | Page equations |
|---|---:|---|
| Appendix title and approximation boundary | 5–8, 30 | GrBP introduction |
| A. Notation and Setup | 13–26 | `pc_setup`, `pc_choices`, `pc_psi`, `pc_sets` |
| A.1 `eq:pt_legacy_prior` | 17–22 | `pc_prior` |
| A.2 `eq:joint_pdf_structured` | 30–40 | `pc_chain` |
| A.3 `eq:pt_counts_distributions` | 43–50 | `pc_poisson`, `pc_cancel` |
| A.4 `eq:I13` | 50–65 | `pc_allocation`, `pc_birthstates` |
| A.5 `eq:pt_group_like_count_split` | 67–77 | `pc_likelihood`, `pc_Llegacy`, `pc_Lnew`, `pc_Fclutter` |
| A.6 `eq:pt_constant_C` | 79–87 | `pc_mu_redistribute`, `pc_C` |
| A.7 `eq:legacy_l_function` | 87–102 | `pc_R`, `pc_legacy` |
| A.8 `eq:new_l_function` | 103–117 | `pc_new` |
| A.9 `eq:final_joint_pdf_local_factors` | 118–128 | `pc_final` |

The presentation uses the explicit shorthands `s=(x,E)`, `B`, `R_j`,
`L_L`, `L_N`, and `F_c`; expanding them recovers the source factors.
The partition is fixed before association. Preserve the Poisson clutter-group
approximation, one inverse `mu_g` per target-generated group, detected-newborn
intensity `mu_n f_n`, and newborn zero truncation. The allocation and likelihood
split is read as the source's joint kernel rather than a separately normalized
association PMF. Splits and cross-target merges are outside this conditional model.

The public addition covers the appendix derivation only; it does not restore
the historical independent grouping model, BP recursion chapters, or toy calculator.

## Factor graph

The final GrBP slide (`#grbp-graph`, slide 47) and the article's
`#grbp-factor-graph` figure render `Drawings/graph_drawing.tex` from the same
manuscript revision. The file exists in the manuscript but is not included
by its current appendix. The unmodified source is `graph-drawing-original.tex`,
SHA-256 `307c379677b4c13f28ebf9a0c02bec7029b2bd44515fac97c6027a7a126f86aa`.

`build-grbp-graph.py` compiles the TikZ and converts it to a self-contained SVG.
Only the count labels change: `n_p` becomes `n^t`, and `n_g` becomes `n^g`,
matching the appendix notation. The time index `k`, graph topology, factor
labels, and prediction paths are retained. The central consistency box
abbreviates the product of pairwise consistency factors. The newborn density
is already inside the newborn local factor; no separate newborn prior is added.
