# Training data policy

AETHRION-LLM must only be trained on data that the project owner has the right to use for machine learning.

Allowed starting points:
- Text written by the project owner.
- Public-domain works, after verifying the jurisdiction and edition.
- Open datasets whose licenses explicitly permit the intended training and redistribution.
- Consented contributions with clear documentation.

Do not add:
- Private messages, personal records, passwords, access tokens, or confidential documents.
- Scraped content where terms or copyright status do not permit training.
- Personal data without a documented lawful basis, necessity, minimization, and appropriate safeguards.
- Evaluation answers copied into the training set.

Keep a manifest for every corpus source: source name, URL or origin, retrieval date, license, permitted uses, preprocessing, and checksum. Remove duplicates and sensitive data. Maintain a held-out evaluation split that is never used for training.
