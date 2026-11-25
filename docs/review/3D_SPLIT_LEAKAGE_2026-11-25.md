# 3D point-cloud split leakage finding — 2026-11-25

## Finding

The 3D pilot contains 210 point clouds, 7 primitive families, 5 shape presets, and 6 generated variants per family/preset pair. There are 35 distinct text captions, each repeated 6 times.

The generator in `data/authored/make_synth_visual.py` currently assigns:

```python
"group_id": f"3d-{s}-{q}-{k // len(combos)}"
```

Because `len(combos)` is 35, each of the six repeated rounds receives a different suffix (0 through 5). This creates 210 distinct group IDs rather than keeping all six variants of the same primitive/preset together.

The preparation pipeline correctly assigns splits using `group_id`, but the generated group IDs are too narrow. Consequently, related point clouds with the same primitive family and same shape preset can be assigned independently to train, validation, and test. This is a split-leakage risk and can make evaluation look better than generalization to unseen shapes warrants.

## Evidence from current authored records

- 210 rows and 210 unique IDs.
- 35 unique text captions.
- Each caption occurs 6 times.
- 7 primitive families × 5 presets = 35 family/preset combinations.
- The generator cycles through those 35 combinations six times and increments the group suffix on every cycle.

## Required correction

1. Define the intended evaluation question first:
   - For unseen instances of known families, group all variants of each family/preset pair together.
   - For unseen primitive families, group by primitive family and hold out entire families. With only seven families, report the small number of groups and uncertainty; do not overstate statistical reliability.
2. Make `group_id` reflect the selected holdout unit, not the generation round. Do not simply hash each record ID.
3. Add tests asserting that no family/preset group (or held-out family, depending on the chosen evaluation) occurs in more than one split.
4. Rebuild the 3D prepared dataset and manifest from the corrected source. Recompute source, point-cloud, NPZ and tokenizer hashes; do not patch the manifest or NPZ by hand.
5. Inspect all 210 NPY arrays for shape, dtype, finite values, bounds, point distribution and geometry-caption consistency. The existing preparation path checks shape, finiteness and numeric bounds when preparing, but that does not establish semantic correctness of the caption.

## Safety and release status

This is a documentation-only audit finding. No records, NPY files, NPZ artifacts, manifests, approval flags, or training settings are changed. The 3D family remains unapproved for training. This report does not certify geometry quality, legal rights, or production 3D generation capability.
