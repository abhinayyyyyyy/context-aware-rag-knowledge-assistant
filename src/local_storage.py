"""
Local Filesystem Object Storage Provider.

Implements BaseObjectStorage for local development and offline workflows.
Ensures path safety, prevents directory traversal, and creates collision-resistant
storage keys while preserving original filename metadata.
"""

import os
from pathlib import Path
import re
import tempfile
from typing import BinaryIO, Dict, Optional, Union
import uuid

from src.storage_config import get_local_storage_dir
from src.storage_interface import BaseObjectStorage


def sanitize_filename(filename: str) -> str:
    """
    Sanitize an uploaded filename to prevent directory traversal and special character exploits.
    Preserves valid file extensions and alphanumeric characters.

    Args:
        filename: Raw filename string from upload.

    Returns:
        Clean, safe filename string.
    """
    if not filename or not isinstance(filename, str):
        return "unnamed_document.pdf"

    # Strip directory components across OS platforms
    basename = os.path.basename(filename.replace("\\", "/")).strip()

    # Remove null bytes, control characters, and leading/trailing whitespace
    clean_base = "".join(c for c in basename if c.isprintable() and c not in "\x00\r\n").strip()

    # Split stem and extension
    stem, ext = os.path.splitext(clean_base)

    # Sanitize stem (allow letters, numbers, hyphens, underscores)
    clean_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem).strip("._")
    if not clean_stem:
        clean_stem = "document"

    # Sanitize extension (default to .pdf if none or empty)
    clean_ext = re.sub(r"[^a-zA-Z0-9]", "", ext).lower()
    if clean_ext:
        clean_ext = f".{clean_ext}"
    else:
        clean_ext = ".pdf"

    return f"{clean_stem}{clean_ext}"


def generate_storage_key(filename: str, prefix: Optional[str] = None) -> str:
    """
    Generate a collision-resistant, provider-neutral storage key.
    Format: [prefix/]<uuid_hex>_<safe_filename>

    Args:
        filename: Original file name.
        prefix: Optional key prefix/folder name.

    Returns:
        Unique storage key string.
    """
    safe_name = sanitize_filename(filename)
    unique_id = uuid.uuid4().hex
    key = f"{unique_id}_{safe_name}"

    if prefix:
        clean_prefix = prefix.strip("/\\ ")
        if clean_prefix:
            return f"{clean_prefix}/{key}"
    return key


class LocalStorageProvider(BaseObjectStorage):
    """
    Local filesystem implementation of BaseObjectStorage.
    Stores files in a local directory with path validation against directory traversal.
    """

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the local storage provider.

        Args:
            storage_dir: Optional directory path. Defaults to get_local_storage_dir().
        """
        raw_dir = storage_dir or get_local_storage_dir()
        self.base_dir = Path(raw_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, storage_key: str) -> Path:
        """
        Resolve storage key to an absolute Path and verify it is strictly
        within the configured base directory.

        Args:
            storage_key: Object storage key.

        Returns:
            Resolved absolute Path.

        Raises:
            ValueError: If storage key is empty, invalid, or traverses outside base directory.
        """
        if not storage_key or not isinstance(storage_key, str):
            raise ValueError("Storage key must be a non-empty string.")

        # Disallow null bytes
        if "\x00" in storage_key:
            raise ValueError(f"Null byte detected in storage key: '{storage_key}'")

        # Disallow explicit traversal indicators
        normalized_str = storage_key.replace("\\", "/")
        parts = normalized_str.split("/")
        if ".." in parts or normalized_str.startswith("/"):
            raise ValueError(f"Path traversal detected in storage key: '{storage_key}'")

        resolved = (self.base_dir / storage_key).resolve()

        # Enforce directory boundary containment
        try:
            resolved.relative_to(self.base_dir)
        except ValueError:
            raise ValueError(
                f"Security violation: storage key '{storage_key}' resolves outside "
                f"storage root '{self.base_dir}'"
            )

        return resolved

    def upload(
        self,
        file_data: Union[bytes, BinaryIO],
        filename: str,
        content_type: str = "application/pdf",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload binary file data to the local storage directory atomically.

        Args:
            file_data: Bytes or binary stream to store.
            filename: Original filename.
            content_type: MIME type (stored in metadata if applicable).
            metadata: Optional metadata dictionary.

        Returns:
            Unique storage key.

        Raises:
            ValueError: If filename or data is invalid.
            RuntimeError: If writing to disk fails.
        """
        if not filename or not isinstance(filename, str):
            raise ValueError("Filename must be a non-empty string.")

        # Read binary bytes
        if hasattr(file_data, "read"):
            data = file_data.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
        elif isinstance(file_data, (bytes, bytearray)):
            data = bytes(file_data)
        else:
            raise ValueError("file_data must be bytes or a readable binary file-like object.")

        if len(data) == 0:
            raise ValueError(f"Cannot upload empty file for '{filename}'.")

        storage_key = generate_storage_key(filename)
        target_path = self._resolve_safe_path(storage_key)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write via temporary file in target directory
        try:
            temp_file = tempfile.NamedTemporaryFile(
                dir=str(target_path.parent),
                delete=False,
                prefix="tmp_upload_",
            )
            temp_path = Path(temp_file.name)
            try:
                temp_file.write(data)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_file.close()
                temp_path.replace(target_path)
            except Exception:
                if temp_path.exists():
                    temp_path.unlink()
                raise
        except Exception as exc:
            raise RuntimeError(f"Failed to write file to local storage for key '{storage_key}': {exc}") from exc

        return storage_key

    def get(self, storage_key: str) -> bytes:
        """
        Retrieve stored file content as bytes.

        Args:
            storage_key: Storage key returned during upload.

        Returns:
            File content bytes.

        Raises:
            FileNotFoundError: If the object does not exist.
            ValueError: If the storage key is invalid or traverses outside root.
        """
        target_path = self._resolve_safe_path(storage_key)
        if not target_path.is_file():
            raise FileNotFoundError(f"Storage object not found for key: '{storage_key}'")

        try:
            return target_path.read_bytes()
        except Exception as exc:
            raise RuntimeError(f"Failed to read storage key '{storage_key}': {exc}") from exc

    def delete(self, storage_key: str) -> bool:
        """
        Delete a stored file by storage key.

        Args:
            storage_key: Storage key of object to delete.

        Returns:
            True if deleted, False if object did not exist.
        """
        target_path = self._resolve_safe_path(storage_key)
        if not target_path.exists():
            return False

        try:
            target_path.unlink()
            return True
        except Exception as exc:
            raise RuntimeError(f"Failed to delete storage key '{storage_key}': {exc}") from exc

    def exists(self, storage_key: str) -> bool:
        """
        Check whether an object exists in local storage.

        Args:
            storage_key: Storage key to verify.

        Returns:
            True if file exists, False otherwise.
        """
        try:
            target_path = self._resolve_safe_path(storage_key)
            return target_path.is_file()
        except ValueError:
            return False
