import pytest

from app.services.parsing import DatasetParseError, parse_dataset

CSV = b"input,output,reference,tone\nhi,Hello!,Hello,warm\nbye,Bye.,,cold\n"


def test_csv_with_reference_and_extra_columns():
    items = parse_dataset("d.csv", CSV)
    assert len(items) == 2
    assert items[0].reference == "Hello"
    assert items[0].metadata == {"tone": "warm"}
    assert items[1].reference is None


def test_jsonl():
    raw = b'{"input": "a", "output": "b"}\n\n{"input": "c", "output": "d", "reference": "e"}\n'
    items = parse_dataset("d.jsonl", raw)
    assert [i.output for i in items] == ["b", "d"]
    assert items[1].reference == "e"
    assert items[0].metadata is None


def test_utf8_bom_is_handled():
    assert len(parse_dataset("d.csv", b"\xef\xbb\xbf" + CSV)) == 2


@pytest.mark.parametrize("raw", [b"", b"   \n"])
def test_empty_file(raw):
    with pytest.raises(DatasetParseError, match="empty"):
        parse_dataset("d.csv", raw)


def test_header_only_csv_has_no_items():
    with pytest.raises(DatasetParseError, match="no items"):
        parse_dataset("d.csv", b"input,output\n")


def test_missing_columns():
    with pytest.raises(DatasetParseError, match="output"):
        parse_dataset("d.csv", b"input,text\nx,y\n")


def test_blank_required_value_names_the_row():
    with pytest.raises(DatasetParseError, match="Item 2: 'output'"):
        parse_dataset("d.csv", b"input,output\na,b\nc,\n")


def test_jsonl_missing_field_and_bad_line():
    with pytest.raises(DatasetParseError, match="Item 1: 'output'"):
        parse_dataset("d.jsonl", b'{"input": "a"}\n')
    with pytest.raises(DatasetParseError, match="Line 2"):
        parse_dataset("d.jsonl", b'{"input": "a", "output": "b"}\nnot json\n')


def test_bad_encoding():
    with pytest.raises(DatasetParseError, match="UTF-8"):
        parse_dataset("d.csv", b"input,output\n\xff\xfe,x\n")


def test_unsupported_extension():
    with pytest.raises(DatasetParseError, match="Unsupported"):
        parse_dataset("d.txt", b"input,output\na,b\n")
