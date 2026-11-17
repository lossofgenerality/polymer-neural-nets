# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
# Authors: Dara, Hoshino
"""Tests of the bibliography (polymer_nn.references): unique keys, a link for
every non-book entry, and the authors and copyright in every rendered form.
"""
from polymer_nn import references as R


def test_keys_unique_and_linked():
    keys = [r.key for r in R.REFERENCES]
    assert len(keys) == len(set(keys))
    for r in R.REFERENCES:
        assert r.topic in R.TOPICS
        assert r.kind == "book" or r.doi or r.url, r.key


def test_rendered_outputs_carry_authors_and_copyright():
    md, bib = R.to_markdown(), R.to_bibtex()
    for text in (md, bib):
        assert "Computational Classnotes" in text and "CC BY-NC-SA 4.0" in text
        assert "Dara" in text and "Hoshino" in text
    assert bib.count("\n@") + bib.startswith("@") == len(R.REFERENCES)
    assert R.cite("satorras2021") == "Satorras, Hoogeboom & Welling (2021)"
