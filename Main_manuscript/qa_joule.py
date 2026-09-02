from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEX = ROOT / "main_manuscript_joule.tex"
REVIEW_TEX = ROOT / "main_manuscript_joule_review.tex"


def latex_plain_text(block: str) -> str:
    block = re.sub(r"(?<!\\)%.*", " ", block)
    block = re.sub(r"\\cite\{[^}]*\}", " ", block)
    for _ in range(3):
        block = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?\{([^{}]*)\}", r" \1 ", block)
    block = re.sub(r"\\[A-Za-z]+|[{}$^_~\\]", " ", block)
    return re.sub(r"\s+", " ", block).strip()


def word_count(block: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-.%][A-Za-z0-9]+)*", latex_plain_text(block)))


def extract(tex: str, pattern: str, label: str, failures: list[str]) -> str:
    match = re.search(pattern, tex, flags=re.S)
    if match is None:
        failures.append(f"Missing {label}.")
        return ""
    return match.group(1).strip()


def main() -> int:
    failures: list[str] = []
    tex = TEX.read_text(encoding="utf-8")
    review = REVIEW_TEX.read_text(encoding="utf-8")

    summary = extract(tex, r"\\begin\{abstract\}(.*?)\\end\{abstract\}", "Summary", failures)
    context = extract(
        tex,
        r"\\section\*\{Context \\& scale\}(.*?)\\section\*\{Highlights\}",
        "Context & scale statement",
        failures,
    )
    highlights = extract(
        tex,
        r"\\section\*\{Highlights\}.*?\\begin\{itemize\}(.*?)\\end\{itemize\}",
        "Highlights",
        failures,
    )
    bullets = [latex_plain_text(item) for item in re.findall(r"\\item\s+(.*?)(?=\\item|$)", highlights, flags=re.S)]

    summary_words = word_count(summary)
    context_chars = len(latex_plain_text(context))
    bullet_chars = [len(item) for item in bullets]

    if summary_words >= 150:
        failures.append(f"Summary has {summary_words} words; Joule requires fewer than 150.")
    if context_chars > 1000:
        failures.append(f"Context & scale has {context_chars} characters; Joule limit is 1,000.")
    if not 3 <= len(bullets) <= 4:
        failures.append(f"Found {len(bullets)} Highlights; Joule permits three or four.")
    for index, length in enumerate(bullet_chars, start=1):
        if length > 85:
            failures.append(f"Highlight {index} has {length} characters; Joule limit is 85.")

    required = [
        r"\\renewcommand\{\\abstractname\}\{Summary\}",
        r"\\section\*\{Introduction\}",
        r"\\section\*\{Results\}",
        r"\\section\*\{Discussion\}",
        r"\\section\*\{Methods\}",
        r"\\section\*\{Resource availability\}",
        r"\\subsection\*\{Lead contact\}",
        r"\\subsection\*\{Materials availability\}",
        r"\\subsection\*\{Data and code availability\}",
        r"\\section\*\{Declaration of interests\}",
        r"\\section\*\{Supplemental information\}",
    ]
    for pattern in required:
        if re.search(pattern, tex) is None:
            failures.append(f"Missing required Joule structure: {pattern}")

    for block_name, pattern in (
        ("Summary", r"\\begin\{abstract\}(.*?)\\end\{abstract\}"),
        ("Context & scale", r"\\section\*\{Context \\& scale\}(.*?)\\section\*\{Highlights\}"),
        ("Highlights", r"\\section\*\{Highlights\}.*?\\begin\{itemize\}(.*?)\\end\{itemize\}"),
    ):
        clean_block = extract(tex, pattern, f"clean {block_name}", failures)
        review_block = extract(review, pattern, f"review {block_name}", failures)
        if clean_block != review_block:
            failures.append(f"Clean and review {block_name} blocks differ.")

    result = {
        "passed": not failures,
        "failures": failures,
        "checks": {
            "summary_words": summary_words,
            "context_scale_characters": context_chars,
            "highlight_characters": bullet_chars,
            "highlight_count": len(bullets),
        },
    }
    (ROOT / "joule_manuscript_qa.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
