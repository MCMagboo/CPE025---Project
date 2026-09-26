"""Local clinician accounts for the OrthoScan AI sign-in screen.

Accounts are kept in config/clinicians.json. Plain-text passwords are never
stored: each account keeps a random salt and a PBKDF2-SHA256 hash.
"""

import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path

ACCOUNTS_FILE = Path(__file__).resolve().parent.parent / "config" / "clinicians.json"
PBKDF2_ITERATIONS = 600_000
MIN_PASSWORD_LENGTH = 8


class AccountError(ValueError):
    """An account could not be created; the message is shown to the user."""


class ClinicianStore:
    """Reads, creates, and verifies clinician accounts on this computer."""

    def __init__(self, path=ACCOUNTS_FILE):
        self.path = Path(path)

    def _load(self):
        if not self.path.exists():
            return {}
        with self.path.open(encoding="utf-8") as f:
            return json.load(f)

    def _save(self, accounts):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file first so a crash can't leave a half-written file.
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(accounts, f, indent=2)
        os.replace(tmp, self.path)

    def has_accounts(self):
        return bool(self._load())

    def create(self, username, display_name, password):
        """Add an account and return its display name."""
        username = username.strip().lower()
        display_name = display_name.strip()
        if not display_name or not username:
            raise AccountError("Enter your full name and a username.")
        if len(password) < MIN_PASSWORD_LENGTH:
            raise AccountError(
                f"Use a password with at least {MIN_PASSWORD_LENGTH} characters."
            )

        accounts = self._load()
        if username in accounts:
            raise AccountError("That username is already taken.")

        salt = secrets.token_bytes(16)
        accounts[username] = {
            "display_name": display_name,
            "salt": salt.hex(),
            "hash": _hash(password, salt, PBKDF2_ITERATIONS).hex(),
            "iterations": PBKDF2_ITERATIONS,
        }
        self._save(accounts)
        return display_name

    def verify(self, username, password):
        """Return the clinician's display name, or None if the credentials are wrong."""
        account = self._load().get(username.strip().lower())
        if account is None:
            # Hash anyway so an unknown username takes as long as a wrong password.
            _hash(password, bytes(16), PBKDF2_ITERATIONS)
            return None

        digest = _hash(password, bytes.fromhex(account["salt"]), account["iterations"])
        if hmac.compare_digest(digest, bytes.fromhex(account["hash"])):
            return account["display_name"]
        return None


def _hash(password, salt, iterations):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
