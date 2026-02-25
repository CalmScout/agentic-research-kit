"""Generic CSV parser for configurable data schemas.

This parser provides format-agnostic CSV ingestion with configurable field mappings,
making it suitable for any CSV-based data source.

Example:
    # For research papers data
    parser = GenericCSVParser(
        field_mappings={
            "content": "abstract",
            "title": "title",
            "metadata": "doi"
        },
        required_columns=["abstract", "title", "doi"]
    )
"""

import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd

logger = logging.getLogger(__name__)


class GenericCSVParser:
    """Format-agnostic CSV parser with configurable field mappings."""

    def __init__(
        self,
        csv_path: str,
        field_mappings: dict[str, str] | None = None,
        required_columns: list[str] | None = None,
        text_columns: list[str] | None = None,
    ):
        """Initialize generic CSV parser.

        Args:
            csv_path: Path to CSV file
            field_mappings: Optional mapping of logical names to CSV column names
                Example: {"content": "text_body", "metadata": "url"}
            required_columns: List of column names that must be present in CSV
            text_columns: Columns containing long text (for special handling)
        """
        self.csv_path = Path(csv_path)
        self.field_mappings = field_mappings or {}
        self.required_columns = required_columns or []
        self.text_columns = text_columns or []

    def load_and_validate(self) -> tuple[pd.DataFrame, dict]:
        """Load CSV with validation.

        Returns:
            Tuple of (DataFrame, statistics_dict)

        Raises:
            ValueError: If required columns are missing
            FileNotFoundError: If CSV file doesn't exist
        """
        # Check if file exists
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        logger.info(f"Loading CSV from {self.csv_path}")

        # Load CSV with error handling for encoding issues
        try:
            df = pd.read_csv(self.csv_path)
        except UnicodeDecodeError:
            logger.warning("UTF-8 decoding failed, trying latin-1")
            df = pd.read_csv(self.csv_path, encoding="latin-1")
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}") from e

        # Validate required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

        # Log file size info
        file_size_mb = self.csv_path.stat().st_size / 1024 / 1024
        logger.info(f"File size: {file_size_mb:.1f}MB")

        # Apply field mappings if provided
        if self.field_mappings:
            df = self._apply_field_mappings(df)

        # Clean data
        df = self._clean_data(df)

        # Generate statistics
        stats = self._generate_statistics(df)

        return df, stats

    def _apply_field_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply field mappings to normalize column names.

        Args:
            df: Raw DataFrame

        Returns:
            DataFrame with normalized column names
        """
        # Create reverse mapping (CSV column -> logical name)
        reverse_mapping = {v: k for k, v in self.field_mappings.items()}

        # Rename columns based on mapping
        renamed_df = df.rename(columns=reverse_mapping)

        # Log mapped columns
        if self.field_mappings:
            logger.info(f"Applied field mappings: {self.field_mappings}")

        return renamed_df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Strip whitespace from string columns
        str_cols = df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            df[col] = df[col].fillna("").astype(str).str.strip()

        # Replace empty strings with NaN for consistency
        df = df.replace("", pd.NA)

        logger.info("Data cleaning complete")
        return df

    def _generate_statistics(self, df: pd.DataFrame) -> dict:
        """Generate dataset statistics.

        Args:
            df: Cleaned DataFrame

        Returns:
            Statistics dictionary
        """
        stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

        # Add text column statistics
        if self.text_columns:
            stats["text_columns"] = {}
            for col in self.text_columns:
                if col in df.columns:
                    text_lengths = df[col].str.len()
                    stats["text_columns"][col] = {
                        "avg_length": float(text_lengths.mean()) if not text_lengths.empty else 0,
                        "max_length": int(text_lengths.max()) if not text_lengths.empty else 0,
                        "min_length": int(text_lengths.min()) if not text_lengths.empty else 0,
                    }

        # Add value counts for categorical columns (low cardinality)
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count <= 20:  # Only for low-cardinality columns
                stats[f"{col}_distribution"] = df[col].value_counts().head(10).to_dict()

        logger.info(
            f"Statistics: {stats['total_rows']} total rows, " f"{stats['total_columns']} columns"
        )

        return stats

    def get_sample_rows(self, n: int = 10) -> list[dict]:
        """Get sample rows for testing.

        Args:
            n: Number of samples to return

        Returns:
            List of row dictionaries
        """
        # Load DataFrame
        df, _ = self.load_and_validate()

        # Sample n rows
        sample_df = df.sample(n=min(n, len(df)), random_state=42)

        # Convert to list of dicts
        samples = sample_df.to_dict("records")

        logger.info(f"Extracted {len(samples)} sample rows")
        return cast(list[dict[str, Any]], samples)

    def get_mapped_field(self, row: dict, logical_field: str, default: Any = None) -> Any:
        """Get field value using field mapping.

        Args:
            row: Data row dictionary
            logical_field: Logical field name (e.g., "content", "metadata")
            default: Default value if field not found

        Returns:
            Field value or default
        """
        # Check if logical field is in mappings
        if logical_field in self.field_mappings:
            csv_column = self.field_mappings[logical_field]
            return row.get(csv_column, default)

        # Otherwise, try direct field name
        return row.get(logical_field, default)
