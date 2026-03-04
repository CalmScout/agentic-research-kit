import os
import tempfile

import pandas as pd
import pytest

from src.data_ingestion.generic_csv_parser import GenericCSVParser


@pytest.fixture
def temp_csv():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        df = pd.DataFrame({
            "col1": ["val1", "val2"],
            "col2": ["val3", "val4"],
            "abstract": ["text1", "text2"],
            "title": ["t1", "t2"],
            "doi": ["d1", "d2"]
        })
        df.to_csv(tf.name, index=False)
        temp_path = tf.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)

def test_csv_parser_load_and_validate(temp_csv):
    parser = GenericCSVParser(
        csv_path=temp_csv,
        required_columns=["col1", "col2"]
    )
    df, stats = parser.load_and_validate()

    assert len(df) == 2
    assert "col1" in df.columns
    assert stats["total_rows"] == 2

def test_csv_parser_missing_required_columns(temp_csv):
    parser = GenericCSVParser(
        csv_path=temp_csv,
        required_columns=["non_existent"]
    )
    with pytest.raises(ValueError, match="Missing required columns"):
        parser.load_and_validate()

def test_csv_parser_field_mappings(temp_csv):
    parser = GenericCSVParser(
        csv_path=temp_csv,
        field_mappings={"content": "abstract", "name": "title"}
    )
    df, _ = parser.load_and_validate()

    # Note: _apply_field_mappings renames columns based on reverse mapping
    # reverse_mapping = {"abstract": "content", "title": "name"}
    assert "content" in df.columns
    assert "name" in df.columns
    assert df["content"].iloc[0] == "text1"

def test_csv_parser_clean_data(temp_csv):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        df = pd.DataFrame({
            "col1": ["  val1  ", ""],
            "col2": [None, "val2"]
        })
        df.to_csv(tf.name, index=False)
        temp_path = tf.name

    parser = GenericCSVParser(csv_path=temp_path)
    df, _ = parser.load_and_validate()

    assert df["col1"].iloc[0] == "val1"
    assert pd.isna(df["col1"].iloc[1])

    if os.path.exists(temp_path):
        os.remove(temp_path)

def test_csv_parser_statistics(temp_csv):
    parser = GenericCSVParser(
        csv_path=temp_csv,
        text_columns=["abstract"]
    )
    _, stats = parser.load_and_validate()

    assert "text_columns" in stats
    assert "abstract" in stats["text_columns"]
    assert stats["text_columns"]["abstract"]["avg_length"] == 5.0

def test_get_sample_rows(temp_csv):
    parser = GenericCSVParser(csv_path=temp_csv)
    samples = parser.get_sample_rows(n=1)

    assert len(samples) == 1
    assert isinstance(samples[0], dict)

def test_get_mapped_field():
    parser = GenericCSVParser(
        csv_path="dummy.csv",
        field_mappings={"content": "abstract"}
    )
    row = {"abstract": "text", "other": "val"}

    assert parser.get_mapped_field(row, "content") == "text"
    assert parser.get_mapped_field(row, "other") == "val"
    assert parser.get_mapped_field(row, "non_existent", "default") == "default"
