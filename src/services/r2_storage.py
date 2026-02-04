"""Cloudflare R2 Storage Service for Exports.

Provides cloud storage for exported CSVs, JSONs, and cached API responses
using Cloudflare R2 (S3-compatible object storage).

Features:
- Upload/download files to R2
- List and delete files
- Optional public URL support
- Automatic content type detection
"""

import json
import logging
import mimetypes
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Try to import boto3 for S3-compatible API
try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.debug("boto3 not available - R2 storage disabled")


class R2Storage:
    """Cloudflare R2 object storage service.

    Uses boto3 with S3-compatible API to interact with Cloudflare R2.
    Supports file upload, download, listing, and deletion.
    """

    def __init__(
        self,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str = "stockpile-exports",
        public_url: Optional[str] = None,
    ):
        """Initialize R2 storage.

        Args:
            account_id: Cloudflare account ID
            access_key_id: R2 API access key ID
            secret_access_key: R2 API secret access key
            bucket_name: R2 bucket name
            public_url: Optional public URL base for files (CDN URL)
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for R2 storage. Install with: pip install boto3")

        self.account_id = account_id
        self.bucket_name = bucket_name
        self.public_url = public_url

        # Configure S3 client for R2
        self._client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )

        logger.info(f"R2 storage initialized for bucket: {bucket_name}")

    def upload_file(
        self,
        key: str,
        data: Union[bytes, str, BinaryIO],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Upload a file to R2.

        Args:
            key: Object key (path in bucket)
            data: File data as bytes, string, or file-like object
            content_type: MIME type (auto-detected if not provided)
            metadata: Optional metadata dict

        Returns:
            Public URL if configured, otherwise S3 URL
        """
        # Convert string to bytes
        if isinstance(data, str):
            data = data.encode("utf-8")

        # Convert bytes to file-like object
        if isinstance(data, bytes):
            data = BytesIO(data)

        # Auto-detect content type from key
        if content_type is None:
            content_type, _ = mimetypes.guess_type(key)
            if content_type is None:
                # Default based on extension
                ext = Path(key).suffix.lower()
                content_type = {
                    ".json": "application/json",
                    ".csv": "text/csv",
                    ".txt": "text/plain",
                    ".html": "text/html",
                    ".xml": "application/xml",
                    ".pdf": "application/pdf",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                }.get(ext, "application/octet-stream")

        extra_args = {"ContentType": content_type}
        if metadata:
            extra_args["Metadata"] = metadata

        try:
            self._client.upload_fileobj(
                data,
                self.bucket_name,
                key,
                ExtraArgs=extra_args,
            )

            logger.info(f"Uploaded {key} to R2")

            # Return appropriate URL
            if self.public_url:
                return f"{self.public_url.rstrip('/')}/{key}"
            else:
                return f"s3://{self.bucket_name}/{key}"

        except ClientError as e:
            logger.error(f"Failed to upload {key}: {e}")
            raise

    def upload_json(
        self,
        key: str,
        data: Union[Dict, List],
        indent: int = 2,
    ) -> str:
        """Upload JSON data to R2.

        Args:
            key: Object key (should end in .json)
            data: Dict or list to serialize as JSON
            indent: JSON indentation level

        Returns:
            URL of uploaded file
        """
        json_bytes = json.dumps(data, indent=indent, default=str).encode("utf-8")
        return self.upload_file(key, json_bytes, content_type="application/json")

    def download_file(self, key: str) -> bytes:
        """Download a file from R2.

        Args:
            key: Object key

        Returns:
            File contents as bytes
        """
        try:
            response = self._client.get_object(Bucket=self.bucket_name, Key=key)
            data = response["Body"].read()
            logger.debug(f"Downloaded {key} from R2 ({len(data)} bytes)")
            return data

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"File not found: {key}")
            logger.error(f"Failed to download {key}: {e}")
            raise

    def download_json(self, key: str) -> Union[Dict, List]:
        """Download and parse JSON from R2.

        Args:
            key: Object key

        Returns:
            Parsed JSON data
        """
        data = self.download_file(key)
        return json.loads(data.decode("utf-8"))

    def list_files(
        self,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> List[Dict]:
        """List files in the bucket.

        Args:
            prefix: Optional prefix to filter files
            max_keys: Maximum number of files to return

        Returns:
            List of file info dicts with key, size, last_modified
        """
        files = []

        try:
            paginator = self._client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=prefix,
                PaginationConfig={"MaxItems": max_keys},
            )

            for page in pages:
                for obj in page.get("Contents", []):
                    files.append({
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"],
                        "etag": obj.get("ETag", "").strip('"'),
                    })

        except ClientError as e:
            logger.error(f"Failed to list files: {e}")
            raise

        return files

    def delete_file(self, key: str) -> bool:
        """Delete a file from R2.

        Args:
            key: Object key

        Returns:
            True if deleted, False if not found
        """
        try:
            self._client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"Deleted {key} from R2")
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return False
            logger.error(f"Failed to delete {key}: {e}")
            raise

    def delete_files(self, keys: List[str]) -> int:
        """Delete multiple files from R2.

        Args:
            keys: List of object keys to delete

        Returns:
            Number of files deleted
        """
        if not keys:
            return 0

        try:
            # R2/S3 supports batch delete of up to 1000 objects
            delete_objects = [{"Key": key} for key in keys[:1000]]
            response = self._client.delete_objects(
                Bucket=self.bucket_name,
                Delete={"Objects": delete_objects},
            )

            deleted = len(response.get("Deleted", []))
            errors = response.get("Errors", [])

            if errors:
                for err in errors:
                    logger.warning(f"Failed to delete {err['Key']}: {err['Message']}")

            logger.info(f"Deleted {deleted} files from R2")
            return deleted

        except ClientError as e:
            logger.error(f"Failed to delete files: {e}")
            raise

    def file_exists(self, key: str) -> bool:
        """Check if a file exists in R2.

        Args:
            key: Object key

        Returns:
            True if file exists, False otherwise
        """
        try:
            self._client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError:
            return False

    def get_presigned_url(
        self,
        key: str,
        expires_in: int = 3600,
        method: str = "get_object",
    ) -> str:
        """Generate a presigned URL for temporary access.

        Args:
            key: Object key
            expires_in: URL expiration time in seconds (default 1 hour)
            method: S3 method ("get_object" for download, "put_object" for upload)

        Returns:
            Presigned URL
        """
        try:
            url = self._client.generate_presigned_url(
                method,
                Params={"Bucket": self.bucket_name, "Key": key},
                ExpiresIn=expires_in,
            )
            return url

        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise

    def get_file_info(self, key: str) -> Optional[Dict]:
        """Get metadata about a file.

        Args:
            key: Object key

        Returns:
            File info dict or None if not found
        """
        try:
            response = self._client.head_object(Bucket=self.bucket_name, Key=key)
            return {
                "key": key,
                "size": response["ContentLength"],
                "content_type": response["ContentType"],
                "last_modified": response["LastModified"],
                "etag": response.get("ETag", "").strip('"'),
                "metadata": response.get("Metadata", {}),
            }
        except ClientError:
            return None


# Factory function
_storage_instance: Optional[R2Storage] = None


def get_r2_storage(
    account_id: Optional[str] = None,
    access_key_id: Optional[str] = None,
    secret_access_key: Optional[str] = None,
    bucket_name: Optional[str] = None,
    public_url: Optional[str] = None,
) -> Optional[R2Storage]:
    """Get or create the R2Storage instance.

    Tries to load credentials from config if not provided.

    Args:
        account_id: Cloudflare account ID
        access_key_id: R2 API access key ID
        secret_access_key: R2 API secret access key
        bucket_name: R2 bucket name
        public_url: Optional public URL base

    Returns:
        R2Storage instance or None if not configured
    """
    global _storage_instance

    if _storage_instance is not None:
        return _storage_instance

    # Try to load from config
    try:
        from utils.config import load_config
        config = load_config()

        account_id = account_id or config.get("r2_account_id")
        access_key_id = access_key_id or config.get("r2_access_key_id")
        secret_access_key = secret_access_key or config.get("r2_secret_access_key")
        bucket_name = bucket_name or config.get("r2_bucket_name", "stockpile-exports")
        public_url = public_url or config.get("r2_public_url")

    except Exception:
        pass

    # Check if all required credentials are provided
    if not all([account_id, access_key_id, secret_access_key]):
        logger.debug("R2 storage not configured - missing credentials")
        return None

    if not BOTO3_AVAILABLE:
        logger.warning("boto3 not installed - R2 storage unavailable")
        return None

    try:
        _storage_instance = R2Storage(
            account_id=account_id,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            bucket_name=bucket_name,
            public_url=public_url,
        )
        return _storage_instance
    except Exception as e:
        logger.error(f"Failed to initialize R2 storage: {e}")
        return None
