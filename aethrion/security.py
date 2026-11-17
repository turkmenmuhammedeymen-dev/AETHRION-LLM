"""Fail-closed bearer authentication for the standalone AETHRION API."""
import os
import secrets

from fastapi import Header, HTTPException


def require_api_token(authorization: str | None = Header(default=None)) -> None:
    """Require a configured API token for every non-health API route."""
    expected = os.environ.get("AETHRION_API_TOKEN", "")
    if not expected or not expected.strip():
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    scheme, separator, supplied = (authorization or "").partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not secrets.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))
    ):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
