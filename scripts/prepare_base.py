"""Deprecated BASE preparer. It must never manufacture a human approval.

The former implementation wrote training_allowed=true and fabricated rights,
privacy, quality and first-party review fields from a hard-coded owner statement.
That behavior was unsafe because no record-level human review was evidenced.
"""
import argparse

def build(force: bool = False) -> dict:
    raise SystemExit(
        "BASE preparation blocked: automatic owner attestation has been disabled. "
        "Review the exact BASE source snapshot and record the evidence first. "
        "No files were changed and no training was started."
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BASE preparation is disabled pending human review.")
    parser.add_argument("--force", action="store_true")
    parser.parse_args()
    build()
