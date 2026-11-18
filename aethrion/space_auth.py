"""Small, dependency-free authentication gate for the Hugging Face Space UI."""


def space_auth_required(auth_enabled: bool) -> bool:
    """Return True when a Space operation must stop before local inference."""
    return auth_enabled is not True
