# AETHRION-SAFE held-out evaluation protocol

**Status: implementation protocol; no trained model or approved test corpus is currently registered.**

The independent metric calculator is `aethrion.safety_metrics.evaluate_safety_predictions`.
It accepts integer labels (`ALLOW=0`, `REVIEW=1`, `BLOCK=2`) and one calibrated
three-class probability row per independently reviewed held-out example.

## Required evaluation design

1. Use only a rights-cleared, privacy-minimized, owner-authored dataset whose
   test split was held out by provenance group before training.
2. Have qualified human reviewers label the test examples independently.
   Record reviewer agreement and adjudicate disagreements outside the model.
3. Keep all test records out of tokenizer fitting, training, threshold tuning,
   and prompt development.
4. Produce probabilities from the exact checkpoint under evaluation. Preserve
   checkpoint, tokenizer, dataset, and evaluation-code SHA-256 values in the
   eventual evaluation report.
5. Evaluate Turkish language variation, benign-versus-harmful intent, ambiguous
   cases, prompt injection in untrusted documents, privacy/exfiltration attempts,
   and category-specific policy behavior. Keep harmful examples abstract and
   non-operational; do not create graphic or actionable harmful content.
6. Report the confusion matrix, per-class precision/recall/F1, BLOCK false
   negative rate, ALLOW-to-BLOCK rate, REVIEW rate/precision/recall, macro-F1,
   and expected calibration error.
7. Review false negatives and false positives with humans. Set release thresholds
   in a separately versioned policy before examining the final test results.

## Metric contract

- At least 25 independently reviewed test examples per decision class are
  required by the calculator. This is only a minimum data sufficiency check,
  not a quality guarantee.
- Low-confidence predictions are routed to REVIEW using the configured
  threshold. The report also records calibration before that abstention step.
- The output status is always `metrics_only_not_release_evidence` and
  `release_approval=false`. Metrics cannot approve data rights, reviewer
  independence, safety robustness, or production release.
- A passing metric report is only one input to a separate human safety review.
  Prompt-injection, privacy, tool authorization, and adversarial testing remain
  independent release gates.

## Current boundary

The repository currently has no approved SAFE production training dataset or
trained SAFE checkpoint. Unit-test fixtures for the metric calculator are
mathematical test inputs only; they are not safety training examples and must
never be added to a production corpus.
