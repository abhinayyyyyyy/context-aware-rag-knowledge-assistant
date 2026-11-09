"""
Database Configuration Module.

Manages connection parameters for PostgreSQL/pgvector vector store.
Ensures database credentials can be configured cleanly without leaking secrets.
"""

import os
import re
from typing import Optional
from urllib.parse import quote_plus, urlsplit, urlunsplit

from dotenv import load_dotenv

load_dotenv()

DEFAULT_POSTGRES_HOST = "localhost"
DEFAULT_POSTGRES_PORT = 5432
DEFAULT_POSTGRES_DB = "rag_assistant"
DEFAULT_POSTGRES_USER = "postgres"


def mask_database_url(url: Optional[str]) -> str:
    """
    Safely mask credentials in a database connection URL.

    Args:
        url: Raw database connection URL.

    Returns:
        Masked URL string where any password is replaced with '****'.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url)
        if parts.password:
            netloc = parts.netloc.replace(f":{parts.password}@", ":****@")
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
        return url
    except Exception:
        # Fallback regex masking
        return re.sub(r":([^:@]+)@", ":****@", url)


def get_database_url() -> str:
    """
    Retrieve the configured PostgreSQL connection URL.
    Checks DATABASE_URL first, then falls back to discrete POSTGRES_* environment variables.

    Returns:
        PostgreSQL connection URL.

    Raises:
        ValueError: If no connection URL or credentials can be resolved.
    """
    raw_url = os.getenv("DATABASE_URL")
    if raw_url and raw_url.strip():
        return raw_url.strip()

    host = os.getenv("POSTGRES_HOST", DEFAULT_POSTGRES_HOST).strip()
    port = os.getenv("POSTGRES_PORT", str(DEFAULT_POSTGRES_PORT)).strip()
    db = os.getenv("POSTGRES_DB", DEFAULT_POSTGRES_DB).strip()
    user = os.getenv("POSTGRES_USER", DEFAULT_POSTGRES_USER).strip()
    password = os.getenv("POSTGRES_PASSWORD", "")

    if password:
        encoded_password = quote_plus(password)
        auth = f"{user}:{encoded_password}"
    else:
        auth = user

    return f"postgresql://{auth}@{host}:{port}/{db}"


def is_persistence_enabled() -> bool:
    """
    Check if PostgreSQL persistence is enabled via PERSISTENCE_ENABLED.
    Defaults to False for safe local execution without database dependencies.
    """
    val = os.getenv("PERSISTENCE_ENABLED", "false").strip().lower()
    return val in ("true", "1", "yes", "enabled")
