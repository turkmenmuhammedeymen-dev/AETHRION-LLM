---
title: AETHRION-LLM
emoji: ✨
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# AETHRION-LLM
Intelligence Without Limits.

Seven capabilities: Chat, Code, Image, 3D, Research, File Analysis and Audio.

Set these under Space Settings → Variables and secrets:
- AETHRION_SPACE_USERNAME and AETHRION_SPACE_PASSWORD to enable authenticated access to paid remote providers
- AETHRION_CODE_API_KEY, AETHRION_CODE_MODEL
- AETHRION_IMAGE_API_KEY or OPENAI_API_KEY
- MESHY_API_KEY
- TAVILY_API_KEY
- AETHRION_AUDIO_API_KEY or OPENAI_API_KEY
- AETHRION_CHECKPOINT and AETHRION_TOKENIZER for the trained AETHRION model

Remote provider adapters stay disabled unless both Space authentication variables
are configured. This prevents an unauthenticated public Space from spending
provider credits through its UI. The app still boots with provider features disabled.

Chat remains unavailable until its matching checkpoint/tokenizer are supplied; no third-party model silently replaces it. 3D jobs are asynchronous. File-analysis inputs are bounded. Project changes are returned as review ZIPs and uploaded code is never executed. Audio is turn-based, not continuous realtime.

Training is not started in this Space. The repository manifest must remain training_allowed=false until the data-license and privacy approvals are complete.
