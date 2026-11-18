# AETHRION Safety Training Curriculum

Status: specification drafted; **no production safety dataset is approved and no safety training has started**.

This curriculum defines two distinct training tracks. It does not replace runtime controls or the independent release review.

## 1. Behaviour expected from every assistant model

AETHRION-BASE and specialist models that produce user-facing language must learn to:

- answer ordinary, benign requests helpfully and accurately;
- refuse requests that would facilitate real-world harm, abuse, exploitation, or unlawful access;
- avoid giving operational instructions that enable violence, weapon use, self-harm, cyber abuse, fraud, privacy violations, or dangerous substance misuse;
- respond calmly and respectfully, without shaming or escalating;
- ask a concise clarification when intent is genuinely ambiguous;
- offer a safe, relevant alternative when one exists;
- protect personal data, credentials, private files, and confidential information;
- distinguish educational, defensive, fictional, and preventive context from actionable harmful intent;
- acknowledge uncertainty instead of inventing facts, permissions, or capabilities.

The BASE safety track is part of instruction/safety fine-tuning and evaluation. It is not the same model as AETHRION-SAFE.

## 2. AETHRION-SAFE classifier

AETHRION-SAFE is a separate reviewer model. Its current output contract is:

- decision: `ALLOW=0`, `REVIEW=1`, `BLOCK=2`;
- reason labels: 16-wide binary multi-hot vector;
- low-confidence predictions must abstain to REVIEW;
- predictions are candidates, not final authorization decisions.

### Proposed reason-label index

| Index | Reason family |
|---:|---|
| 0 | benign / ordinary assistance |
| 1 | violence or weapon facilitation |
| 2 | self-harm or suicide facilitation |
| 3 | cyber abuse or unauthorized access |
| 4 | malware, credential theft, or account compromise |
| 5 | fraud, impersonation, or financial deception |
| 6 | privacy invasion, doxxing, or sensitive-data exposure |
| 7 | child exploitation or child endangerment |
| 8 | sexual exploitation or coercion |
| 9 | harassment, threats, or targeted abuse |
| 10 | dangerous substances or poisoning |
| 11 | evasion of safety, law, or accountability controls |
| 12 | extremist recruitment or violent propaganda |
| 13 | high-impact advice requiring qualified review |
| 14 | ambiguous intent / insufficient context |
| 15 | other policy-sensitive or unclassified case |

This taxonomy is a proposed annotation contract. A human reviewer must approve or revise it before any production labels are created. Multi-hot reasons describe relevant risk families; they do not independently determine the decision.

## 3. Dataset construction rules

- Use only first-party, project-authored examples for this project. Do not import Common Voice, ShapeNet, CodeSearchNet, LibriSpeech, public safety corpora, or third-party model outputs.
- Create balanced benign, borderline, and disallowed-intent examples. Include safe educational, defensive, prevention, and fictional-context examples so the model does not learn to block entire topics.
- For each example, record a stable sample ID, group ID, split, language, scenario family, input text, expected decision, reason labels, short reviewer rationale, author/reviewer roles, and review timestamp.
- Do not include real people's private messages, personal records, credentials, secrets, or identifying details.
- Do not put operational harmful instructions into examples. Represent disallowed requests at a high level and pair them with a safe response target where applicable.
- Keep related paraphrases, templates, and scenario variants in the same group and the same split. Deduplicate before splitting.
- Use group-disjoint train, validation, and test sets. Keep the test set held out from training and threshold selection.
- Human review is required for every production label. The dataset author cannot silently self-approve the review record.
- Audio examples require speaker consent; this text safety track does not authorize audio collection.

## 4. Evaluation before release

Evaluate on a held-out, group-disjoint set and report at minimum:

- per-class precision, recall, F1, and confusion matrix;
- false-ALLOW rate on disallowed examples;
- false-BLOCK rate on benign examples;
- REVIEW/abstention rate and coverage;
- calibration and performance by language and scenario family;
- robustness to paraphrases, misspellings, prompt injection, quoted text, and conflicting instructions;
- regression results for safe educational, defensive, accessibility, and age-appropriate requests.

Thresholds must be selected on validation data, frozen before test evaluation, and approved by a human reviewer. A model must not be released based on aggregate accuracy alone.

## 5. Runtime defence in depth

AETHRION-SAFE is not a standalone safety boundary. Production must also enforce deterministic input/output checks, prompt-injection separation, authentication, authorization, rate limits, least privilege, secret isolation, safe file handling, audit logging, and human escalation for high-impact cases. IMAGE and 3D outputs need separate modality-specific safety verification.

## 6. Current gate and next evidence required

The training gate remains closed until the following are real and reviewed:

1. first-party authored dataset files and source inventory;
2. documented authorship and rights-holder role;
3. explicit first-party origin attestation for the complete dataset;
4. timestamped independent human safety-label review;
5. privacy and quality review;
6. deduplication and group-disjoint split report;
7. dataset hashes and schema validation;
8. successful training, held-out evaluation, checkpoint integrity, and release approval.

A project-wide permission statement is not a substitute for these evidence records. Never change `training_allowed` to true merely to bypass missing evidence. Synthetic CI fixtures remain test-only and must never be treated as production training data.
