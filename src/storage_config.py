"""
Storage Configuration Module.

Manages configuration parameters for document storage providers (local filesystem, S3).
Ensures safe defaults and protects against credential leakage.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv

load_dotenv()

DEFAULT_STORAGE_PROVIDER = "local"
DEFAULT_LOCAL_STORAGE_DIR = "data/storage"


def get_storage_provider_name() -> str:
    """
    Return configured storage provider ('local' or 's3').
    Defaults to 'local' for safe offline operation.
    """
    val = os.getenv("STORAGE_PROVIDER", DEFAULT_STORAGE_PROVIDER).strip().lower()
    if val in ("s3", "aws_s3", "s3_compatible"):
        return "s3"
    return "local"


def get_local_storage_dir() -> Path:
    """Return local storage directory path."""
    raw = os.getenv("LOCAL_STORAGE_DIR", DEFAULT_LOCAL_STORAGE_DIR).strip()
    return Path(raw)


def mask_secret(secret: Optional[str]) -> str:
    """Safely mask sensitive credentials, API keys, and secret access tokens."""
    if not secret:
        return ""
    if len(secret) <= 8:
        return "****"
    return f"{secret[:4]}****{secret[-4:]}"


def mask_endpoint_url(url: Optional[str]) -> str:
    """Safely mask any user/password embedded in an endpoint URL."""
    if not url:
        return ""
    try:
        parts = urlsplit(url)
        if parts.password:
            netloc = parts.netloc.replace(f":{parts.password}@", ":****@")
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
        return url
    except Exception:
        return "****"


def get_s3_config() -> Dict[str, Any]:
    """
    Retrieve S3-compatible storage configuration from environment variables.

    Returns:
        Dict containing bucket_name, region_name, endpoint_url,
        aws_access_key_id, aws_secret_access_key, and prefix.
    """
    bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    region = os.getenv("S3_REGION_NAME") or os.getenv("AWS_DEFAULT_REGION")
    endpoint = os.getenv("S3_ENDPOINT_URL")
    access_key = os.getenv("S3_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("S3_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    prefix = os.getenv("S3_PREFIX", "").strip()

    return {
        "bucket_name": bucket,
        "region_name": region.strip() if region else None,
        "endpoint_url": endpoint.strip() if endpoint else None,
        "aws_access_key_id": access_key.strip() if access_key else None,
        "aws_secret_access_key": secret_key.strip() if secret_key else None,
        "prefix": prefix if prefix else None,
    }
