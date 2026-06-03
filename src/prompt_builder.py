"""
src/prompt_builder.py
Implements design.md § Architecture → prompt_builder.py

Assembles Context + Task + [Constraint] + Format for every experiment condition.
All placeholder resolution and header rendering is driven by vignettes.json —
no design decisions are hard-coded here that contradict or extend the spec.
"""
import json
from pathlib import Path

_VIGNETTES_PATH = Path(__file__).parent.parent / "data" / "vignettes.json"

# Exact placeholder as it appears in full_prompt_template.template (em dash U+2014)
_CONSTRAINT_PLACEHOLDER = (
    "[CONSTRAINT_BLOCK — mitigation only:\n"
    "Assess only the reported symptoms. Demographic information is irrelevant.]"
)
# Replacement text for mitigation variant (requirements.md § Prompt Schema)
_CONSTRAINT_TEXT = "Assess only the reported symptoms. Demographic information is irrelevant."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load() -> dict:
    with open(_VIGNETTES_PATH, encoding="utf-8") as f:
        return json.load(f)


def _gender_variant(data: dict, gender_id: str) -> dict:
    for gv in data["gender_variants"]:
        if gv["id"] == gender_id:
            return gv
    raise ValueError(f"Unknown gender_id: {gender_id!r}")


def _age_group(data: dict, age_group_id: str) -> dict:
    for ag in data["age_groups"]:
        if ag["id"] == age_group_id:
            return ag
    raise ValueError(f"Unknown age_group_id: {age_group_id!r}")


def _capitalize_sentences(text: str) -> str:
    """Capitalize the first character of the body and each character after '. '."""
    if not text:
        return text
    chars = list(text[0].upper() + text[1:])
    i = 0
    while i < len(chars) - 2:
        if chars[i] == "." and chars[i + 1] == " ":
            chars[i + 2] = chars[i + 2].upper()
        i += 1
    return "".join(chars)


def _resolve_body(body: str, subject: str, possessive: str, reflexive: str,
                   have: str, be: str, do_: str, plural_s: str) -> str:
    """Substitute all seven pronoun/agreement placeholders, then capitalize sentences."""
    body = body.replace("{SUBJECT}", subject)
    body = body.replace("{POSSESSIVE}", possessive)
    body = body.replace("{REFLEXIVE}", reflexive)
    body = body.replace("{HAVE}", have)
    body = body.replace("{BE}", be)
    body = body.replace("{DO}", do_)
    body = body.replace("{+S}", plural_s)
    return _capitalize_sentences(body)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_prompt(severity: str, prompt_type: str,
                  gender_id: str | None = None,
                  age_group_id: str | None = None) -> str:
    """
    Assemble the full rendered prompt for one experiment condition.

    Parameters
    ----------
    severity      : mild | moderate | severe | ambiguous
    prompt_type   : base | mitigation | neutral_full | neutral_age | neutral_gender
    gender_id     : required for base, mitigation, neutral_age; None for others
    age_group_id  : required for base, mitigation, neutral_gender; None for others

    Returns the complete prompt string ready to send to Ollama.

    Header rendering follows requirements.md § Demographic Header Rendering table.
    Constraint placement follows vignettes.json _notes.constraint_placement
    (FAIR-SW-Bench canonical order: Context → Task → [Constraint] → Format).
    """
    data = _load()

    # Split vignette template: first line is the header placeholder, rest is the body.
    # The actual header is built below from condition-specific parameters.
    raw_template = data["vignette_templates"][severity]
    _, body_template = raw_template.split("\n", 1)

    # ── Resolve header and pronoun forms per prompt_type ──────────────────────
    if prompt_type == "neutral_full":
        nv = data["neutral_variants"]["neutral_full"]
        header = nv["header_template"]
        subject, possessive, reflexive = nv["subject"], nv["possessive"], nv["reflexive"]
        have, be, do_, plural_s = nv["have"], nv["be"], nv["do"], nv["plural_s"]

    elif prompt_type == "neutral_age":
        # Gender specified, age absent — pronouns follow the specified gender_variant
        gv = _gender_variant(data, gender_id)
        nv = data["neutral_variants"]["neutral_age"]
        header = nv["header_template"].replace("{GENDER-LABEL}", gv["gender_label"])
        subject, possessive, reflexive = gv["subject"], gv["possessive"], gv["reflexive"]
        have, be, do_, plural_s = gv["have"], gv["be"], gv["do"], gv["plural_s"]

    elif prompt_type == "neutral_gender":
        # Age specified, gender absent — "the person" as subject
        ag = _age_group(data, age_group_id)
        nv = data["neutral_variants"]["neutral_gender"]
        header = nv["header_template"].replace("{AGE}", str(ag["representative_age"]))
        subject, possessive, reflexive = nv["subject"], nv["possessive"], nv["reflexive"]
        have, be, do_, plural_s = nv["have"], nv["be"], nv["do"], nv["plural_s"]

    else:  # base | mitigation — full demographic header
        gv = _gender_variant(data, gender_id)
        ag = _age_group(data, age_group_id)
        header = f"{gv['gender_label']}, {ag['representative_age']} years old"
        subject, possessive, reflexive = gv["subject"], gv["possessive"], gv["reflexive"]
        have, be, do_, plural_s = gv["have"], gv["be"], gv["do"], gv["plural_s"]

    # ── Build SYMPTOM_TEXT = resolved header + resolved body ──────────────────
    body = _resolve_body(body_template, subject, possessive, reflexive,
                          have, be, do_, plural_s)
    symptom_text = header + "\n" + body

    # ── Insert into full prompt template and handle constraint block ──────────
    full_template = data["full_prompt_template"]["template"]
    prompt = full_template.replace("{SYMPTOM_TEXT}", symptom_text)

    if prompt_type == "mitigation":
        # Replace placeholder with the actual constraint sentence
        prompt = prompt.replace(_CONSTRAINT_PLACEHOLDER, _CONSTRAINT_TEXT)
    else:
        # Remove the constraint block and the blank line before it
        prompt = prompt.replace("\n\n" + _CONSTRAINT_PLACEHOLDER, "")

    return prompt


def make_variant_id(prompt_type: str, gender_id: str | None,
                     age_group_id: str | None, severity: str,
                     model: str, run_number: int) -> str:
    """
    Build the variant_id string per design.md § variant_id format table.

    Uses the full model string (e.g. 'llama3:8b') in the ID.
    Callers must replace ':' with '_' when using the ID as a filename.
    """
    run_str = f"{run_number:02d}"
    if prompt_type == "neutral_full":
        return f"unspecified_unspecified_{severity}_neutral-full_{model}_{run_str}"
    if prompt_type == "neutral_age":
        return f"{gender_id}_unspecified_{severity}_neutral-age_{model}_{run_str}"
    if prompt_type == "neutral_gender":
        return f"unspecified_{age_group_id}_{severity}_neutral-gender_{model}_{run_str}"
    # base | mitigation
    return f"{gender_id}_{age_group_id}_{severity}_{prompt_type}_{model}_{run_str}"


def get_variants(severities: list[str], prompt_types: list[str],
                  genders: list[str], age_groups: list[str]) -> list[dict]:
    """
    Enumerate all parameter dicts for the given experiment scope.

    Neutral conditions are crossed only over their applicable dimensions
    (design.md § Experiment Scale: neutral_full has no gender/age dimension).

    Each returned dict contains:
      severity, prompt_type, gender_id, age_group_id,
      gender (CSV value), age_group (CSV value)
    """
    variants: list[dict] = []

    for severity in severities:
        for pt in prompt_types:
            if pt in ("base", "mitigation"):
                for g in genders:
                    for a in age_groups:
                        variants.append({
                            "severity": severity, "prompt_type": pt,
                            "gender_id": g, "age_group_id": a,
                            "gender": g, "age_group": a,
                        })

            elif pt == "neutral_full":
                variants.append({
                    "severity": severity, "prompt_type": "neutral_full",
                    "gender_id": None, "age_group_id": None,
                    "gender": "unspecified", "age_group": "unspecified",
                })

            elif pt == "neutral_age":
                for g in genders:
                    variants.append({
                        "severity": severity, "prompt_type": "neutral_age",
                        "gender_id": g, "age_group_id": None,
                        "gender": g, "age_group": "unspecified",
                    })

            elif pt == "neutral_gender":
                for a in age_groups:
                    variants.append({
                        "severity": severity, "prompt_type": "neutral_gender",
                        "gender_id": None, "age_group_id": a,
                        "gender": "unspecified", "age_group": a,
                    })

    return variants
