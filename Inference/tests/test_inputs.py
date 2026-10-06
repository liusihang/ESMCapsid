import gzip

import pytest

from esmcapsid.inputs import read_sequences


def test_ids_do_not_depend_on_header_collisions(tmp_path):
    path = tmp_path / "input.faa"
    path.write_text(">A\nACD\n>A\nMNP\n>A__dup002\nAAA\n>seq_000000001\nGGG\n")
    records = read_sequences(path)
    assert len({record.internal_id for record in records}) == 4
    assert [record.original_id for record in records] == ["A", "A", "A__dup002", "seq_000000001"]
    assert [record.index for record in records] == [1, 2, 3, 4]


def test_gzip_cleaning_and_invalid_records(tmp_path):
    path = tmp_path / "input.faa.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(">first full header\n acd eF*\n>empty\n>gap\nAC-D\n>interior_stop\nAC*D\n")
    records = read_sequences(path)
    assert records[0].sequence == "ACDEF"
    assert records[0].cleaned_terminal_stop
    assert records[0].header == "first full header"
    assert records[1].issue == "empty_sequence"
    assert records[2].issue == "invalid_characters:-"
    assert records[3].issue == "invalid_characters:*"


def test_csv_columns_are_explicit_and_no_label_is_needed(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text('name,protein\na,ACD\na,MNP\n')
    records = read_sequences(path, "protein", "name")
    assert [record.sequence for record in records] == ["ACD", "MNP"]
    with pytest.raises(ValueError, match="column"):
        read_sequences(path)


@pytest.mark.parametrize("contents", ["", "ACD\n>A\nAAA"])
def test_invalid_fasta_fails_before_models_are_loaded(tmp_path, contents):
    path = tmp_path / "input.faa"
    path.write_text(contents)
    with pytest.raises(ValueError):
        read_sequences(path)
