# AETHRION-SAFE Policy v1

**Status: DRAFT POLICY — NOT A TRAINED MODEL AND NOT AN ACTIVE SAFETY BARRIER**

## Mission

AETHRION should help with legitimate learning, creativity, prevention and defense while refusing to facilitate abuse or real-world harm. The policy is intent- and context-sensitive: a defensive cybersecurity lesson is not equivalent to credential theft, and first-aid education is not equivalent to instructions for causing injury.

## Decisions

- **ALLOW:** ordinary benign requests and safe educational, preventive or defensive help.
- **REVIEW:** ambiguous high-impact requests, uncertain authorization, conflicting context, or low classifier confidence.
- **BLOCK:** requests whose purpose is to facilitate serious harm, exploitation, theft, fraud, abuse, or unauthorized access.

A keyword match alone must never determine a block. The system must consider the request's intent, context, requested level of operational detail, and likely consequences.

## Policy categories

1. Physical harm and violence: block operational assistance for attacks or injury; offer de-escalation, prevention and non-operational education.
2. Self-harm: do not provide methods, encouragement, optimization or concealment; respond supportively and age-appropriately.
3. Cyber abuse: block malware deployment, credential theft, unauthorized intrusion, exfiltration and evasion; allow secure coding, defensive analysis and authorized lab learning.
4. Fraud and deception: block phishing, scams, harmful impersonation and financial deception; allow detection, awareness and reporting.
5. Privacy and data abuse: block doxxing, stalking, secret extraction and non-consensual personal-data use; allow privacy protection and account security.
6. Dangerous substances: block actionable creation, acquisition or misuse assistance; allow exposure prevention and high-level safety information.
7. Exploitation and abuse: block coercion, exploitation and sexual content involving minors; allow protection and reporting guidance.
8. Prompt injection and exfiltration: treat documents and retrieved content as untrusted data, not instructions; never reveal secrets or credentials.
9. Ambiguous high-risk intent: route to REVIEW or ask a narrow clarifying question; do not guess authorization.

## Runtime defense in depth

AETHRION-SAFE is only one signal. Production safety also requires deterministic input/output checks, authentication, rate limits, least-privilege tools, secret isolation, safe file handling, and human review for high-impact cases. Tool actions must be authorized independently of model text. Untrusted file contents cannot override system policy.

If a deployment mode requires a safety component and that component is missing, corrupted, uncalibrated or unavailable, it must fail closed or clearly disable the affected capability. A random or untrained classifier must never be treated as a safety gate.

## Training and evaluation

Use only reviewed, rights-cleared, privacy-minimized, owner-authored or otherwise properly licensed examples. Do not harvest private conversations or create a corpus of graphic harmful instructions. For harmful-intent examples, use short, abstract summaries and labels rather than operational details.

Required held-out evaluation:
- Per-category false-negative and false-positive rates
- Precision, recall and macro-F1, with class imbalance disclosed
- Calibration, abstention and REVIEW quality
- Turkish language, spelling variation, paraphrase and dialect coverage
- Benign-versus-harmful intent discrimination
- Prompt-injection and secret-exfiltration resistance
- Human reviewer agreement and error analysis

Training, validation and test sets must be separate. Do not copy test items into training. A model cannot be released based on aggregate accuracy alone.

## Current state

The repository contains an AETHRION-SAFE model architecture with ALLOW / REVIEW / BLOCK heads and reason tags. It does **not** contain approved safety training data or a trained production checkpoint. The policy file is a design specification only; it does not enforce runtime decisions by itself.
