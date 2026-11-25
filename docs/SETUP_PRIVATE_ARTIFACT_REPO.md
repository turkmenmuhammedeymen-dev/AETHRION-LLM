# Owner setup: private artifact repo for Space training (5 minutes)

1. Hugging Face -> New Model -> name e.g. `aethrion-artifacts`, visibility **Private**.
2. Hugging Face -> Settings -> Access Tokens -> create a **fine-grained** token limited to that one repo (write).
3. Space -> Settings -> Variables and secrets:
   - Secret `HF_TOKEN` = the token (enter it only in the HF UI; never paste it in chat or commit it).
   - Variable `AETHRION_ARTIFACT_REPO` = `<your-user>/aethrion-artifacts`.
4. Restart the Space. `aethrion/artifact_sync.py` refuses to run if the repo is public.
5. Confirm the Space runtime reports both the secret and variable as configured; a missing secret blocks private artifact pull/push even if the Space health check is green.
6. Data approval: complete `aethrion-human-review-v1` for each model after reading `docs/review/*_REVIEW.md`.
Training still needs BASE (tokenizer + pretrained weights) before CHAT; see docs/ZEROGPU_SESSION_TRAINING.md.
