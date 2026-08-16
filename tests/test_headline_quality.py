"""Headline quality guards + LLM column reassignment in StorySummarizer."""
import json
from types import SimpleNamespace

from structlog.testing import capture_logs

from src.synthesis.summarizer import (
    COLUMNS,
    MAX_HEADLINE_LEN,
    ORDERING_SYSTEM_PROMPT,
    StorySummarizer,
)


class _StubLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate(self, prompt, system_prompt=None, max_tokens=None,
                 response_format=None, **kwargs):
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response_format": response_format,
        })
        return json.dumps(self.payload)


class _Cluster:
    source_names = ["alpha", "beta"]
    source_count = 2
    representative_text = "Corroborated detail " * 6


class _FakeGraph:
    corroborated = [_Cluster() for _ in range(3)]

    def to_viz_dict(self):
        return {
            "corroborated": [
                {"sources": [{"text": f"Passage {i} " + "word " * 15,
                              "source_name": f"Src{i}"}]}
                for i in range(3)
            ],
            "unique_details": [],
        }


def _story(column="politics"):
    return SimpleNamespace(
        id="story-1",
        headline="Original cluster headline",
        coverage_spread={"center": 3},
        articles=[
            {"body": "body text " * 10, "source_slug": f"src{i}",
             "url": f"https://example.com/{i}", "column": column,
             "source_bias": "center"}
            for i in range(3)
        ],
    )


def _ordering(headline, column=None):
    payload = {
        "headline": headline,
        "ordering": [
            {"cluster": 0, "transition": ""},
            {"cluster": 1, "transition": "Meanwhile..."},
            {"cluster": 2, "transition": "Later..."},
        ],
    }
    if column is not None:
        payload["column"] = column
    return payload


def _synthesize(payload, column="politics"):
    llm = _StubLLM(payload)
    result = StorySummarizer(llm).synthesize(_story(column), _FakeGraph())
    assert result is not None
    return result, llm


# -- schema: headline maxLength + column enum -------------------------------


def test_schema_has_headline_maxlength_and_column_enum():
    _, llm = _synthesize(_ordering("Valid headline", column="sports"))
    schema = llm.calls[0]["response_format"]["json_schema"]["schema"]
    props = schema["properties"]
    assert props["headline"]["maxLength"] == 90
    assert props["column"]["enum"] == list(COLUMNS)
    assert props["column"]["type"] == "string"
    assert "column" not in schema["required"]


def test_prompt_states_current_section_and_headline_rules():
    _, llm = _synthesize(_ordering("Valid headline"))
    prompt = llm.calls[0]["prompt"]
    assert "Current section: politics." in prompt
    assert '"column"' in prompt
    system = llm.calls[0]["system_prompt"]
    assert "12 words" in system
    assert "semicolon" in system.lower()
    assert "one story" in system.lower()


# -- headline post-processing -----------------------------------------------


def test_semicolon_mashup_trimmed_to_first_segment():
    mashup = ("Tommy John Dies at 83; First Player Hits Three Home Runs "
              "in Three At-Bats")
    with capture_logs() as logs:
        result, _ = _synthesize(_ordering(mashup))
    assert result.generated_headline == "Tommy John Dies at 83"
    assert any(e["event"] == "headline_trimmed" for e in logs)


def test_overlong_headline_truncated_at_word_boundary():
    words = [f"word{i:02d}" for i in range(24)]  # 6-char words, 167 joined
    long_headline = " ".join(words)
    assert len(long_headline) > MAX_HEADLINE_LEN
    result, _ = _synthesize(_ordering(long_headline))
    got = result.generated_headline
    assert len(got) <= MAX_HEADLINE_LEN
    assert got in long_headline  # cut at a word boundary, not mid-word
    assert " " in got  # not a bare hard slice
    expected_prefix = " ".join(words[:15])  # 15 words = 104 chars <= 110
    assert got == expected_prefix


def test_clean_headline_unchanged_when_fine():
    result, _ = _synthesize(_ordering("Senate passes funding bill"))
    assert result.generated_headline == "Senate passes funding bill"


# -- column reassignment -----------------------------------------------------


def test_valid_different_column_lands_in_to_dict():
    with capture_logs() as logs:
        result, _ = _synthesize(_ordering("Navy crew rescued at sea",
                                          column="money"),
                                column="politics")
    assert result.column == "money"
    assert result.to_dict()["column"] == "money"
    reassigned = [e for e in logs if e["event"] == "story_column_reassigned"]
    assert reassigned and reassigned[0]["from"] == "politics"
    assert reassigned[0]["to"] == "money"


def test_same_column_absent_from_to_dict():
    result, _ = _synthesize(_ordering("Valid headline", column="politics"))
    assert result.column is None
    assert "column" not in result.to_dict()


def test_invalid_column_absent_from_to_dict():
    result, _ = _synthesize(_ordering("Valid headline", column="world"))
    assert result.column is None
    assert "column" not in result.to_dict()
