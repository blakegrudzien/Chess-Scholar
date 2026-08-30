import pytest

from src.ingestion.hash_utils import check_no_delimiter


def test_check_no_delimiter_passes_clean_fields():
    check_no_delimiter("Alice", "Bob", "e4 d5")  # must not raise


def test_check_no_delimiter_rejects_a_field_containing_the_delimiter():
    with pytest.raises(ValueError, match=r"\|"):
        check_no_delimiter("Alice", "Weird|Name", "e4 d5")


def test_check_no_delimiter_accepts_zero_fields():
    check_no_delimiter()  # must not raise
