"""Unit tests for TTS preprocessing in VideoProductionAgent."""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from video_agent.agent import VideoProductionAgent, _number_to_words, _year_to_words


@pytest.mark.unit
class TestNumberToWords:
    def test_basic_numbers(self):
        assert _number_to_words(0) == "zero"
        assert _number_to_words(5) == "five"
        assert _number_to_words(13) == "thirteen"
        assert _number_to_words(42) == "forty-two"
        assert _number_to_words(100) == "one hundred"
        assert _number_to_words(500) == "five hundred"

    def test_large_numbers(self):
        assert "billion" in _number_to_words(6_000_000_000)
        assert "million" in _number_to_words(1_000_000)
        assert "thousand" in _number_to_words(5_000)


@pytest.mark.unit
class TestYearToWords:
    def test_common_years(self):
        assert _year_to_words(1987) == "nineteen eighty-seven"
        assert _year_to_words(2000) == "two thousand"
        assert _year_to_words(2024) == "twenty twenty-four"
        assert _year_to_words(1900) == "nineteen hundred"


@pytest.mark.unit
class TestPreprocessForTTS:
    preprocess = staticmethod(VideoProductionAgent._preprocess_for_tts)

    def test_number_conversion(self):
        result = self.preprocess("There are 5 items.")
        assert "five" in result
        assert "5" not in result

    def test_year_conversion(self):
        result = self.preprocess("In 1987 it happened.")
        assert "nineteen eighty-seven" in result
        assert "1987" not in result

    def test_abbreviation_expansion(self):
        result = self.preprocess("AI is here.")
        assert "A.I." in result

    def test_abbreviation_multiple(self):
        result = self.preprocess("The CEO of the FBI.")
        assert "C.E.O." in result
        assert "F.B.I." in result

    def test_em_dash_replacement(self):
        result = self.preprocess("word\u2014another.")
        assert "\u2014" not in result
        assert "," in result

    def test_ampersand_replacement(self):
        result = self.preprocess("bread & butter.")
        assert "and" in result
        assert "&" not in result

    def test_terminal_punctuation(self):
        result = self.preprocess("Hello world")
        assert result.endswith(".")

    def test_paralinguistic_tags_preserved(self):
        result = self.preprocess("[laugh] That's funny.")
        assert "[laugh]" in result

    def test_currency(self):
        result = self.preprocess("worth $500 million.")
        assert "$" not in result
        assert "five hundred" in result
        assert "million" in result
        assert "dollars" in result

    def test_percentage(self):
        result = self.preprocess("grew 15%.")
        assert "fifteen percent" in result
        assert "%" not in result

    def test_slash_to_or(self):
        result = self.preprocess("win/lose situation.")
        assert "win or lose" in result
        assert "/" not in result

    def test_already_clean_passthrough(self):
        clean = "This sentence is already clean."
        result = self.preprocess(clean)
        assert result == clean

    def test_tag_stripping_regex(self):
        regex = VideoProductionAgent._PARALINGUISTIC_TAG_RE
        assert regex.search("[laugh]")
        assert regex.search("[sigh]")
        assert regex.search("[gasp]")
        assert not regex.search("[random]")
        text = "[laugh] Hello [sigh] world"
        cleaned = regex.sub("", text).strip()
        assert "[laugh]" not in cleaned
        assert "[sigh]" not in cleaned
