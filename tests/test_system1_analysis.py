import pytest

from backend.config.loader import universe_config
from backend.system1_news.keywords import extract_keywords
from backend.system1_news.relevance import CompanyRef, detect_entities
from backend.system1_news.sentiment import LexiconSentimentEngine, label_for

REFS = [CompanyRef(c["ticker"], c["name"], c["sector"], c["aliases"], c.get("case_sensitive_aliases", []),
                   c.get("exclude_patterns", [])) for c in universe_config()["companies"]]
ENGINE = LexiconSentimentEngine()


def ents(title, summary=None):
    return {e.entity: e for e in detect_entities(title, summary, REFS)}


def test_company_identification_and_category():
    e = ents("HDFC Bank reports strong quarterly profit growth")
    assert set(e) == {"HDFCBANK"}
    assert e["HDFCBANK"].category == "profit"
    assert e["HDFCBANK"].relevance >= 0.7


def test_excluded_subsidiaries_are_not_the_parent():
    assert "HDFCBANK" not in ents("HDFC Life shares jump after strong premium growth")
    assert "RELIANCE" not in ents("Reliance Power wins solar contract")
    assert "SBIN" not in ents("SBI Card reports higher spends")


def test_case_sensitive_short_aliases():
    assert "ITC" in ents("ITC to raise cigarette prices")
    assert "ITC" not in ents("the itch to buy stocks")
    assert "LT" in ents("L&T bags mega order from Saudi Arabia")


def test_macro_news_detected():
    e = ents("RBI changes repo rate, holds inflation outlook")
    assert "MACRO" in e and e["MACRO"].category == "interest_rate"


def test_unrelated_news_ignored():
    assert ents("Local football club wins regional cup") == {}


def test_market_wrap_lowers_per_company_relevance():
    single = ents("Infosys shares rise")["INFY"].relevance
    many = ents("Infosys, TCS, HDFC Bank, ICICI Bank and Reliance Industries lead gains")["INFY"].relevance
    assert many < single


def test_keyword_extraction_includes_taxonomy_and_free_terms():
    kws = extract_keywords("Reliance Jio announces spectrum auction win; net profit rises")
    assert "net profit" in kws
    assert any("spectrum" in k for k in kws)


def test_sentiment_direction_and_bounds():
    pos = ENGINE.score("HDFC Bank profit surges, beats estimates", None).score
    neg = ENGINE.score("Infosys shares plunge after weak guidance and downgrade", None).score
    assert pos > 30 and neg < -30
    assert -100 <= neg <= 100 and -100 <= pos <= 100
    assert ENGINE.score("Company holds annual general meeting", None).score == 0


def test_inverted_objects_and_negation():
    assert ENGINE.score("Losses widen at the telecom unit", None).score < 0
    assert ENGINE.score("RBI announces rate cut", None).score > 0
    assert ENGINE.score("Inflation eases in August", None).score > 0
    assert ENGINE.score("Inflation rises sharply", None).score < 0
    assert ENGINE.score("Profit does not rise", None).score < 0


@pytest.mark.parametrize("headline,sign", [
    ("Retail inflation at 16-month high of 3.9% as food items get dearer", -1),
    ("India's April-May fiscal deficit at Rs 1.62 lakh crore, widens on-year", -1),
    ("Sensex slips on FII outflow amid market downturn", -1),
    ("ITC Q1 results: Net profit falls 27%, misses estimates", -1),
    ("Union Budget: FM seeks to lower debt to fund priority sectors", 1),
    ("HDFC Bank shares hit 52-week high after strong results", 1),
])
def test_real_headline_regressions(headline, sign):
    s = ENGINE.score(headline, None).score
    assert s * sign > 15, (headline, s)


def test_label_thresholds():
    assert label_for(16) == "POSITIVE" and label_for(-16) == "NEGATIVE" and label_for(10) == "NEUTRAL"
