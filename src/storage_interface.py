"""
Object Storage Interface.

Defines the abstract base class for object storage backends, ensuring
decoupling between document processing/persistence and underlying storage
providers (e.g., local filesystem, AWS S3, MinIO, Cloudflare R2).
"""

from abc import ABC, abstractmethod
from typing import BinaryIO, Dict, Optional, Union


class BaseObjectStorage(ABC):
    """
    Abstract interface for object storage providers.
    Enables swapping between local filesystem and S3-compatible cloud storage
    without impacting ingestion, retrieval, or persistence logic.
    """

    @abstractmethod
    def upload(
        self,
        file_data: Union[bytes, BinaryIO],
        filename: str,
        content_type: str = "application/pdf",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload binary file data and return a unique, provider-neutral storage key.

        Args:
            file_data: Binary file data as bytes or a readable binary file-like object.
            filename: Original filename (used for metadata and safe extension preservation).
            content_type: MIME type of the uploaded object. Defaults to 'application/pdf'.
            metadata: Optional user-defined metadata dictionary.

        Returns:
            A unique, provider-neutral storage key string.

        Raises:
            ValueError: If file data or filename is invalid.
            RuntimeError: If the upload operation fails.
        """
        pass

    @abstractmethod
    def get(self, storage_key: str) -> bytes:
        """
        Retrieve stored object content as bytes.

        Args:
            storage_key: Unique object identifier.

        Returns:
            Binary content of the stored object as bytes.

        Raises:
            FileNotFoundError: If the object does not exist.
            ValueError: If the storage key is invalid or contains traversal sequences.
            RuntimeError: If retrieval fails.
        """
        pass

    @abstractmethod
    def delete(self, storage_key: str) -> bool:
        """
        Delete a stored object by its storage key.

        Args:
            storage_key: Unique object identifier.

        Returns:
            True if the object was deleted, False if it did not exist.

        Raises:
            ValueError: If the storage key is invalid.
            RuntimeError: If deletion fails due to provider errors.
        """
        pass

    @abstractmethod
    def exists(self, storage_key: str) -> bool:
        """
        Check whether an object exists for the given storage key.

        Args:
            storage_key: Unique object identifier.

        Returns:
            True if object exists, False otherwise.

        Raises:
            ValueError: If the storage key is invalid.
        """
        pass
