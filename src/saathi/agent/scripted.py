"""Deterministic, $0 replacements for the two LLM roles in the arm-5 conversation eval.

- template_opening(): the user's first message, stating EXACTLY the revealed facts, in
  English, Hindi (Devanagari) or Hinglish (Roman script).
- ScriptedSimulator: answers the agent's clarifying question truthfully from the hidden
  profile, one template sentence per field asked. It uses the field names the agent
  attached to its question; if there are none, it falls back to keywords in the question
  text. A field the user has no value for gets "not sure".

Limitation (stated in the README): template sentences are cleaner than real users, so
this measures whether the agent asks for and uses the right facts, not robustness to
messy language. The LLM simulator stays available with --simulator llm.
"""

import random

from saathi.llm.base import Usage
from saathi.schemas import Language

SCRIPTED_USAGE = dict(
    requested_model="scripted",
    served_model="scripted",
    input_tokens=0,
    output_tokens=0,
    latency_s=0.0,
    cost_usd=0.0,
)

OCCUPATIONS = {
    Language.HI: {
        "farmer": "किसान",
        "student": "छात्र",
        "teacher": "शिक्षक",
        "street_vendor": "रेहड़ी-पटरी विक्रेता",
        "unemployed": "बेरोज़गार",
        "construction_worker": "निर्माण मज़दूर",
    },
    Language.HINGLISH: {
        "farmer": "kisan",
        "street_vendor": "rehdi-patri wala",
        "unemployed": "berozgar",
        "construction_worker": "construction mazdoor",
    },
}
GENDER = {
    Language.EN: {"female": "a woman", "male": "a man", "other": "of another gender"},
    Language.HI: {"female": "महिला", "male": "पुरुष", "other": "अन्य लिंग"},
    Language.HINGLISH: {"female": "mahila", "male": "purush", "other": "other gender"},
}
AREA = {
    Language.EN: {"rural": "in a village", "urban": "in a city"},
    Language.HI: {"rural": "गाँव में", "urban": "शहर में"},
    Language.HINGLISH: {"rural": "gaon mein", "urban": "shehar mein"},
}


def _occ(value: str, lang: Language) -> str:
    return OCCUPATIONS.get(lang, {}).get(value, value.replace("_", " "))


def _num(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def _yes_no(lang: Language, value: bool, en: tuple, hi: tuple, hg: tuple) -> str:
    table = {Language.EN: en, Language.HI: hi, Language.HINGLISH: hg}[lang]
    return table[0] if value else table[1]


def fact_sentence(field: str, value, lang: Language) -> str:
    """One sentence stating one profile fact."""
    L = lang
    match field:
        case "age":
            return {
                Language.EN: f"I am {value} years old.",
                Language.HI: f"मेरी उम्र {value} साल है।",
                Language.HINGLISH: f"Meri umar {value} saal hai.",
            }[L]
        case "gender":
            return {
                Language.EN: f"I am {GENDER[L][value]}.",
                Language.HI: f"मैं {GENDER[L][value]} हूँ।",
                Language.HINGLISH: f"Main {GENDER[L][value]} hoon.",
            }[L]
        case "annual_family_income_inr":
            return {
                Language.EN: f"Our family income is Rs {value} per year.",
                Language.HI: f"हमारे परिवार की सालाना आय ₹{value} है।",
                Language.HINGLISH: f"Hamari family ki saalana income {value} rupaye hai.",
            }[L]
        case "state":
            return {
                Language.EN: f"I live in {value}.",
                Language.HI: f"मेरा राज्य {value} है।",
                Language.HINGLISH: f"Mera state {value} hai.",
            }[L]
        case "area":
            return {
                Language.EN: f"I live {AREA[L][value]}.",
                Language.HI: f"मेरा घर {AREA[L][value]} है।",
                Language.HINGLISH: f"Mera ghar {AREA[L][value]} hai.",
            }[L]
        case "occupation":
            return {
                Language.EN: f"I work as a {_occ(value, L)}.",
                Language.HI: f"मैं {_occ(value, L)} हूँ।",
                Language.HINGLISH: f"Main {_occ(value, L)} hoon.",
            }[L]
        case "land_hectares":
            return {
                Language.EN: f"I have {_num(value)} hectares of land.",
                Language.HI: f"मेरे पास {_num(value)} हेक्टेयर ज़मीन है।",
                Language.HINGLISH: f"Mere paas {_num(value)} hectare zameen hai.",
            }[L]
        case "social_category":
            v = str(value).upper()
            return {
                Language.EN: f"My category is {v}.",
                Language.HI: f"मेरी श्रेणी {v} है।",
                Language.HINGLISH: f"Meri category {v} hai.",
            }[L]
        case "is_income_tax_payer":
            return _yes_no(
                L,
                value,
                ("I pay income tax.", "I do not pay income tax."),
                ("हाँ, मैं आयकर दाता हूँ।", "नहीं, मैं आयकर दाता नहीं हूँ।"),
                ("Haan, main income tax payer hoon.", "Nahi, main income tax payer nahi hoon."),
            )
        case "has_bank_account":
            return _yes_no(
                L,
                value,
                ("I have a bank account.", "I do not have a bank account."),
                ("मेरा बैंक खाता है।", "मेरा बैंक खाता नहीं है।"),
                ("Mera bank account hai.", "Mera bank account nahi hai."),
            )
        case "is_government_employee":
            return _yes_no(
                L,
                value,
                ("I am a government employee.", "I am not a government employee."),
                ("मैं सरकारी कर्मचारी हूँ।", "मैं सरकारी कर्मचारी नहीं हूँ।"),
                ("Main sarkari naukri mein hoon.", "Main sarkari naukri mein nahi hoon."),
            )
    raise ValueError(f"no template for field {field!r}")


NOT_SURE = {
    Language.EN: "I am not sure about that.",
    Language.HI: "मुझे इसके बारे में पक्का नहीं पता।",
    Language.HINGLISH: "Iske baare mein pakka nahi pata.",
}
OPENINGS = {
    Language.EN: ["Hi. {facts}Am I eligible for {scheme}?", "{facts}Can I get {scheme}?"],
    Language.HI: [
        "नमस्ते। {facts}क्या मैं {scheme} के लिए पात्र हूँ?",
        "{facts}क्या मुझे {scheme} का लाभ मिल सकता है?",
    ],
    Language.HINGLISH: [
        "Namaste. {facts}Kya main {scheme} ke liye eligible hoon?",
        "{facts}Kya mujhe {scheme} ka fayda mil sakta hai?",
    ],
}

# Fallback when the agent's question lists no field names: keywords in its text.
KEYWORDS = {
    "age": ["age", "old", "उम्र", "आयु", "umar", "umr", "saal"],
    "gender": ["gender", "woman", "man", "लिंग", "महिला", "mahila"],
    "annual_family_income_inr": ["income", "earn", "आय", "आमदनी", "kamai", "aamdani"],
    "state": ["state", "राज्य", "rajya"],
    "area": ["village", "city", "rural", "urban", "गाँव", "शहर", "gaon", "shehar"],
    "occupation": ["occupation", "work", "job", "profession", "काम", "पेशा", "kaam", "naukri"],
    "land_hectares": ["land", "hectare", "acre", "ज़मीन", "जमीन", "zameen", "jameen"],
    "social_category": ["category", "caste", "श्रेणी", "जाति", "jaati"],
    "is_income_tax_payer": ["tax", "कर"],
    "has_bank_account": ["bank", "account", "बैंक", "खाता", "khata"],
    "is_government_employee": ["government employee", "सरकारी", "sarkari"],
}


def fields_from_text(question: str) -> list[str]:
    q = question.lower()
    return [f for f, words in KEYWORDS.items() if any(w in q for w in words)]


def template_opening(
    scheme_name: str, language: Language, facts: dict, rng: random.Random | None = None
) -> tuple[str, Usage]:
    rng = rng or random.Random(0)
    stated = " ".join(fact_sentence(f, v, language) for f, v in facts.items())
    text = rng.choice(OPENINGS[language]).format(
        facts=f"{stated} " if stated else "", scheme=scheme_name
    )
    return text, Usage(**SCRIPTED_USAGE)


class ScriptedSimulator:
    """Answers clarifying questions from the hidden profile with template sentences."""

    uses_fields = True

    def reply(self, scenario, question: str, fields: list[str] | None = None) -> tuple[str, Usage]:
        asked = list(fields or []) or fields_from_text(question)
        facts = {k: v for k, v in scenario.true_profile.model_dump().items() if v is not None}
        parts = [fact_sentence(f, facts[f], scenario.language) for f in asked if f in facts]
        if not parts or any(f not in facts for f in asked):
            parts.append(NOT_SURE[scenario.language])
        return " ".join(parts), Usage(**SCRIPTED_USAGE)
