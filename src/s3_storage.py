"""
S3-Compatible Object Storage Provider.

Implements BaseObjectStorage for cloud deployments using AWS S3, MinIO,
Cloudflare R2, or any S3-compatible object storage API.
Keeps credentials optional for local development and securely masks secrets.
"""

from typing import Any, BinaryIO, Dict, Optional, Union

from src.local_storage import generate_storage_key, sanitize_filename
from src.storage_config import mask_endpoint_url, mask_secret
from src.storage_interface import BaseObjectStorage


class S3StorageProvider(BaseObjectStorage):
    """
    S3-compatible implementation of BaseObjectStorage.
    Interfaces with AWS S3, MinIO, LocalStack, and Ceph using boto3.
    """

    def __init__(
        self,
        bucket_name: str,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        prefix: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        """
        Initialize the S3 storage provider.

        Args:
            bucket_name: S3 bucket name.
            region_name: Optional AWS region (e.g., 'us-east-1').
            endpoint_url: Optional custom endpoint URL for MinIO/R2/LocalStack.
            aws_access_key_id: Optional access key ID.
            aws_secret_access_key: Optional secret access key.
            prefix: Optional key prefix/folder name.
            client: Optional pre-configured boto3 S3 client (used for testing and mocking).

        Raises:
            ValueError: If bucket_name is empty.
            ImportError: If boto3 is not installed and no client is provided.
        """
        if not bucket_name or not isinstance(bucket_name, str):
            raise ValueError("S3 bucket_name must be a non-empty string.")

        self.bucket_name = bucket_name.strip()
        self.region_name = region_name
        self.endpoint_url = endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.prefix = prefix

        if client is not None:
            self.client = client
        else:
            try:
                import boto3
            except ImportError as err:
                raise ImportError(
                    "boto3 is required for S3StorageProvider. "
                    "Install it via 'pip install boto3' or add it to requirements.txt."
                ) from err

            client_kwargs: Dict[str, Any] = {}
            if self.region_name:
                client_kwargs["region_name"] = self.region_name
            if self.endpoint_url:
                client_kwargs["endpoint_url"] = self.endpoint_url
            if self.aws_access_key_id and self.aws_secret_access_key:
                client_kwargs["aws_access_key_id"] = self.aws_access_key_id
                client_kwargs["aws_secret_access_key"] = self.aws_secret_access_key

            self.client = boto3.client("s3", **client_kwargs)

    def upload(
        self,
        file_data: Union[bytes, BinaryIO],
        filename: str,
        content_type: str = "application/pdf",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload binary file data to the configured S3 bucket.

        Args:
            file_data: Bytes or binary stream to store.
            filename: Original filename.
            content_type: MIME type of the uploaded object.
            metadata: Optional user metadata dictionary.

        Returns:
            Provider-neutral storage key.

        Raises:
            ValueError: If file data or filename is invalid.
            RuntimeError: If the S3 upload operation fails.
        """
        if not filename or not isinstance(filename, str):
            raise ValueError("Filename must be a non-empty string.")

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

        storage_key = generate_storage_key(filename, prefix=self.prefix)
        safe_meta = dict(metadata or {})
        safe_meta["original_filename"] = sanitize_filename(filename)

        try:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=storage_key,
                Body=data,
                ContentType=content_type,
                Metadata=safe_meta,
            )
            return storage_key
        except Exception as exc:
            safe_endpoint = mask_endpoint_url(self.endpoint_url) if self.endpoint_url else "AWS S3"
            raise RuntimeError(
                f"Failed to upload '{filename}' to S3 bucket '{self.bucket_name}' ({safe_endpoint}): {exc}"
            ) from exc

    def get(self, storage_key: str) -> bytes:
        """
        Retrieve stored object from S3 as bytes.

        Args:
            storage_key: S3 object key.

        Returns:
            Object binary content.

        Raises:
            FileNotFoundError: If the object does not exist.
            RuntimeError: If retrieval fails.
        """
        if not storage_key or not isinstance(storage_key, str):
            raise ValueError("Storage key must be a non-empty string.")

        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=storage_key,
            )
            body = response["Body"]
            return body.read()
        except Exception as exc:
            error_code = ""
            if hasattr(exc, "response") and isinstance(exc.response, dict):
                error_code = str(exc.response.get("Error", {}).get("Code", ""))

            if error_code in ("NoSuchKey", "404", "NotFound"):
                raise FileNotFoundError(
                    f"Object not found in S3 bucket '{self.bucket_name}': '{storage_key}'"
                ) from exc
            raise RuntimeError(
                f"Failed to retrieve object '{storage_key}' from S3 bucket '{self.bucket_name}': {exc}"
            ) from exc

    def exists(self, storage_key: str) -> bool:
        """
        Check whether an object exists in the S3 bucket.

        Args:
            storage_key: S3 object key.

        Returns:
            True if object exists, False otherwise.
        """
        if not storage_key or not isinstance(storage_key, str):
            return False

        try:
            self.client.head_object(
                Bucket=self.bucket_name,
                Key=storage_key,
            )
            return True
        except Exception as exc:
            error_code = ""
            if hasattr(exc, "response") and isinstance(exc.response, dict):
                error_code = str(exc.response.get("Error", {}).get("Code", ""))

            if error_code in ("404", "NoSuchKey", "NotFound"):
                return False
            # For network/auth failures, bubble up as error
            raise RuntimeError(
                f"Error checking existence for key '{storage_key}' in S3 bucket '{self.bucket_name}': {exc}"
            ) from exc

    def delete(self, storage_key: str) -> bool:
        """
        Delete an object from the S3 bucket.

        Args:
            storage_key: S3 object key to delete.

        Returns:
            True if deleted, False if object did not exist.
        """
        if not storage_key or not isinstance(storage_key, str):
            raise ValueError("Storage key must be a non-empty string.")

        if not self.exists(storage_key):
            return False

        try:
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=storage_key,
            )
            return True
        except Exception as exc:
            raise RuntimeError(
                f"Failed to delete object '{storage_key}' from S3 bucket '{self.bucket_name}': {exc}"
            ) from exc
