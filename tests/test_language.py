import pytest

from saathi.agent.language import detect_language
from saathi.schemas import Language


@pytest.mark.parametrize(
    "text,expected",
    [
        ("What documents do I need for PM-KISAN?", Language.EN),
        ("Am I eligible for the pension scheme? I am 35.", Language.EN),
        ("मुझे पीएम किसान योजना के लिए कौन से दस्तावेज़ चाहिए?", Language.HI),
        ("PM-KISAN में कितना पैसा मिलता है?", Language.HI),
        ("Mujhe PM-KISAN ka paisa kitna milega?", Language.HINGLISH),
        ("Meri umar 35 saal hai, kya main eligible hoon?", Language.HINGLISH),
        ("kitna milega?", Language.HINGLISH),
        ("I have 2 acres of land", Language.EN),
        ("Main farmer hoon", Language.HINGLISH),  # 'main' + short message
        ("", Language.EN),
    ],
)
def test_detect_language(text, expected):
    assert detect_language(text) == expected
