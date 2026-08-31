from pathlib import Path
import re
import unicodedata

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
TEX = (
    ROOT / "supplementary_information_nature_article.tex"
    if (ROOT / "supplementary_information_nature_article.tex").is_file()
    else ROOT / "supplementary_information.tex"
)
PDF = TEX.with_suffix(".pdf")
CURRENT_TITLE = "Curtailment traps early green hydrogen assets beyond the reach of learning"
HAN_PATTERN = re.compile(r"[\u3400-\u9fff]")

reader = PdfReader(str(PDF))
pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)


def normalise_pdf_text(text: str) -> str:
    """Collapse layout whitespace and compatibility ligatures before matching."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


normalised_pdf_text = normalise_pdf_text(pdf_text)

text_extensions = {".tex", ".md", ".txt", ".csv", ".json", ".py"}
files_with_han = []
for path in ROOT.rglob("*"):
    if path.is_file() and path.suffix.lower() in text_extensions:
        content = path.read_text(encoding="utf-8-sig", errors="replace")
        if HAN_PATTERN.search(content):
            files_with_han.append(path.relative_to(ROOT).as_posix())

figure_pdfs = sorted((ROOT / "figures").glob("*.pdf"))
figures_with_han = []
for figure in figure_pdfs:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(figure)).pages)
    if HAN_PATTERN.search(text):
        figures_with_han.append(figure.name)

checks = {
    "pages": len(reader.pages),
    "characters": len(pdf_text),
    "has_current_title": CURRENT_TITLE in normalised_pdf_text,
    "has_primary_1809": "1,809" in pdf_text,
    "has_primary_1099": "1,099" in pdf_text,
    "has_primary_710": "710" in pdf_text,
    "has_30_year_primary_label": "30-year" in normalised_pdf_text,
    "unqualified_old_1889_lines": [
        line.strip()
        for line in TEX.read_text(encoding="utf-8").splitlines()
        if "1,889" in line and "G16" not in line
    ],
    "has_unresolved_markers": "??" in pdf_text,
    "pdf_contains_chinese": bool(HAN_PATTERN.search(pdf_text)),
    "source_files_with_chinese": files_with_han,
    "figure_pdfs_with_chinese": figures_with_han,
}

failures = []
if not checks["has_current_title"]:
    failures.append("The Supplementary Information title does not match the main manuscript.")
if not all(
    checks[key]
    for key in (
        "has_primary_1809",
        "has_primary_1099",
        "has_primary_710",
        "has_30_year_primary_label",
    )
):
    failures.append(
        "Current 30-year headline cohort values are absent from the compiled Supplementary Information."
    )
if checks["unqualified_old_1889_lines"]:
    failures.append(
        "The obsolete 1,889 cohort value appears without an explicit G16 sensitivity qualifier."
    )
if checks["has_unresolved_markers"]:
    failures.append("The compiled Supplementary Information contains unresolved references.")
if checks["pdf_contains_chinese"] or files_with_han or figures_with_han:
    failures.append("Chinese characters remain in the English-only submission package.")

for key, value in checks.items():
    print(f"{key}={value}")
if failures:
    for failure in failures:
        print(f"FAIL: {failure}")
    raise SystemExit(1)
print("passed=True")
