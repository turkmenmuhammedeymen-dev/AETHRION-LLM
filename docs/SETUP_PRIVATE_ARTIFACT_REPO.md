# Owner setup: private artifact repo for Space training (5 minutes)

1. Hugging Face -> Settings -> Access Tokens -> create a **fine-grained** token with write access limited to the private `vizyon2027/aethrion-artifacts` model repo.
2. Space -> Settings -> Variables and secrets:
   - Secret `HF_TOKEN` = that fine-grained token (enter it only in the HF UI).
   - Secret `AETHRION_SPACE_USERNAME` and secret `AETHRION_SPACE_PASSWORD` = the Gradio login pair used by the owner-only session panel.
   - Secret `AETHRION_BASE_AUDIT_HMAC_KEY` = a separately generated random key of at least 32 bytes; required only when preparing CHAT/BASE's signed corpus audit.
   - Variable `AETHRION_ARTIFACT_REPO` = `vizyon2027/aethrion-artifacts`.
3. GitHub repository -> Settings -> Secrets and variables -> Actions: configure `HF_TOKEN`, `AETHRION_SPACE_USERNAME` and `AETHRION_SPACE_PASSWORD` for deployment/session workflows. Configure `AETHRION_BASE_AUDIT_HMAC_KEY` there too if using the runner-based BASE preparation workflow. Use the matching Space login pair, but do not reuse the HF token as a password.
4. Restart the Space, then use its setup panel to confirm authentication, `HF_TOKEN` and `AETHRION_ARTIFACT_REPO` are present. A green health check alone does not prove artifact sync works.
5. Complete and document each model's human data review after checking the source files, provenance, rights, privacy, quality and (for audio) speaker consent. Owner approval is not a substitute for reviewing each source.
Training still needs BASE (tokenizer + pretrained weights) before CHAT; see docs/ZEROGPU_SESSION_TRAINING.md.
