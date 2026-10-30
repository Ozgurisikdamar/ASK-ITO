from app.ingest.clean import clean_extract, parse_sections, plain_text

SAMPLE = """LIGO is a large-scale physics experiment.

It detects gravitational waves.

== History ==
Construction began in 1994.

== Mathematics ==
The strain is {\\displaystyle h=\\Delta L/L} a tiny number.

== See also ==
Virgo interferometer

== References ==
1. Some citation.

== External links ==
Official website
"""


def test_parse_sections_splits_on_headers():
    secs = parse_sections(SAMPLE)
    titles = [s.title for s in secs]
    assert titles[0] == ""  # lead
    assert "History" in titles
    assert "References" in titles


def test_clean_drops_boilerplate_sections():
    secs = clean_extract(SAMPLE)
    titles = {s.title for s in secs}
    assert "References" not in titles
    assert "External links" not in titles
    assert "See also" not in titles
    assert "History" in titles


def test_clean_strips_displaystyle_math():
    secs = clean_extract(SAMPLE)
    math = next(s for s in secs if s.title == "Mathematics")
    assert "\\displaystyle" not in math.text
    assert "h=" in math.text.replace(" ", "")


def test_plain_text_includes_headers():
    secs = clean_extract(SAMPLE)
    text = plain_text(secs)
    assert "== History ==" in text
    assert "gravitational waves" in text
