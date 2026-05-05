from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Optional

import requests

LOGGER = logging.getLogger("github_integration")
GITHUB_API = "https://api.github.com"


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def get_pr_files(repo_full_name: str, pr_number: int, token: str) -> list:
    url = f"{GITHUB_API}/repos/{repo_full_name}/pulls/{pr_number}/files"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_file_content(
    repo_full_name: str, path: str, ref: str, token: str
) -> Optional[str]:
    url = f"{GITHUB_API}/repos/{repo_full_name}/contents/{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.raw+json",
    }
    try:
        resp = requests.get(url, headers=headers, params={"ref": ref}, timeout=15)
        if resp.status_code == 200:
            return resp.text
    except Exception as exc:
        LOGGER.warning("Impossible de récupérer %s@%s : %s", path, ref, exc)
    return None


def post_pr_comment(
    repo_full_name: str, pr_number: int, body: str, token: str
) -> None:
    url = f"{GITHUB_API}/repos/{repo_full_name}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    resp = requests.post(url, headers=headers, json={"body": body}, timeout=15)
    resp.raise_for_status()
    LOGGER.info("Commentaire posté sur PR #%d", pr_number)
