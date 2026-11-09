"""
Object Storage Factory.

Instantiates configured BaseObjectStorage implementations ('local' or 's3').
Ensures clean decoupling between storage configuration and consumers.
"""

from typing import Optional

from src.local_storage import LocalStorageProvider
from src.s3_storage import S3StorageProvider
from src.storage_config import (
    get_local_storage_dir,
    get_s3_config,
    get_storage_provider_name,
)
from src.storage_interface import BaseObjectStorage


def create_storage_provider(provider_type: Optional[str] = None) -> BaseObjectStorage:
    """
    Factory function to create a BaseObjectStorage provider instance.

    Args:
        provider_type: Optional explicit provider identifier ('local' or 's3').
            If None, reads from STORAGE_PROVIDER environment variable.

    Returns:
        BaseObjectStorage provider instance.

    Raises:
        ValueError: If provider_type is unsupported or required configuration is missing.
    """
    resolved_type = (provider_type or get_storage_provider_name()).strip().lower()

    if resolved_type == "local":
        return LocalStorageProvider(storage_dir=get_local_storage_dir())

    if resolved_type in ("s3", "aws_s3", "s3_compatible"):
        cfg = get_s3_config()
        if not cfg["bucket_name"]:
            raise ValueError(
                "S3StorageProvider requires 'S3_BUCKET_NAME' environment variable to be configured."
            )
        return S3StorageProvider(
            bucket_name=cfg["bucket_name"],
            region_name=cfg["region_name"],
            endpoint_url=cfg["endpoint_url"],
            aws_access_key_id=cfg["aws_access_key_id"],
            aws_secret_access_key=cfg["aws_secret_access_key"],
            prefix=cfg["prefix"],
        )

    raise ValueError(
        f"Unsupported storage provider: '{resolved_type}'. Supported providers are: 'local', 's3'."
    )


# Alias matching factory naming in other modules
get_storage_provider = create_storage_provider
