"""Deprecated: automatic owner attestation is disabled.

A statement string or session identifier does not prove that record rights,
provenance, privacy, quality, safety, or speaker consent have been reviewed.
This command performs no writes and cannot change approval flags.
"""
def main() -> None:
    raise SystemExit(
        "Automatic dataset attestation is disabled. This command cannot approve rights, "
        "privacy, quality, safety, or speaker consent. No files were changed."
    )

if __name__ == "__main__":
    main()
