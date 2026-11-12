"""Build the submission zip i25-8007_Assignment2_DS-A.zip.

    python package_submission.py          # only the 13 files required by the brief
    python package_submission.py --full   # + code, tests, figures, outputs (= the repository)

Layout (required items of the brief first; source code, tests, figures and
outputs are added so the zip is runnable and mirrors the GitHub repository):

i25-8007_Assignment2_DS-A/
|-- i25-8007_Assignment2_DS-A.ipynb   (all cells executed)
|-- report.pdf
|-- embeddings/ tfidf_matrix.npy ppmi_matrix.npy embeddings_w2v.npy word2idx.json
|-- models/     bilstm_pos.pt bilstm_ner.pt transformer_cls.pt
|-- data/       pos_*.conll ner_*.conll (+ A1 inputs, part2_sentences.json)
|-- src/ tests/ tools/ scripts/ figures/ outputs/ README.md requirements.txt build_*.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402

REQUIRED = [
    f"{C.SUBMISSION_STEM}.ipynb", "report.pdf",
    "embeddings/tfidf_matrix.npy", "embeddings/ppmi_matrix.npy",
    "embeddings/embeddings_w2v.npy", "embeddings/word2idx.json",
    "models/bilstm_pos.pt", "models/bilstm_ner.pt", "models/transformer_cls.pt",
    "data/pos_train.conll", "data/pos_test.conll", "data/ner_train.conll", "data/ner_test.conll",
]
EXTRA_DIRS = ["src", "tests", "tools", "scripts", "figures", "outputs", "data"]
EXTRA_FILES = ["README.md", "requirements.txt", "build_notebook.py", "build_report.py",
               "notebook_analysis.json", "report_text.json",
               "package_submission.py", ".gitignore"]
SKIP = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def main() -> None:
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        sys.exit(f"missing required files: {missing}")
    files = list(REQUIRED)
    if "--full" not in sys.argv:             # default: only the files the brief lists
        EXTRA_DIRS.clear()
        EXTRA_FILES.clear()
    for d in EXTRA_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP):
                files.append(str(p.relative_to(ROOT)))
    files += [f for f in EXTRA_FILES if (ROOT / f).exists()]
    files = list(dict.fromkeys(files))
    out = ROOT / f"{C.SUBMISSION_STEM}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(ROOT / f, f"{C.SUBMISSION_STEM}/{f}")
    print(f"wrote {out.name}: {len(files)} files, {out.stat().st_size / 1e6:.1f} MB")
    for f in REQUIRED:
        print("  ", f)


if __name__ == "__main__":
    main()
