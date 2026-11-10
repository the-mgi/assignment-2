"""Central paths, identity and hyper-parameters for Assignment 2.

Every script and the notebook import from here, so the numbers that the
assignment brief fixes (d = 100, k = 5, K = 10, ...) live in exactly one place.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Identity
# ----------------------------------------------------------------------------
STUDENT_NAME = "Muhammad Usama"
STUDENT_ID = "i25-8007"
SECTION = "DS-A"
COURSE = "DS 5007: Natural Language Processing"
REPO_NAME = f"{STUDENT_ID}-NLP-Assignment2"
# Filled in once the public GitHub repository exists (see README.md).
REPO_URL = f"https://github.com/<github-username>/{REPO_NAME}"
SUBMISSION_STEM = f"{STUDENT_ID}_Assignment2_{SECTION}"

# ----------------------------------------------------------------------------
# Paths (the layout mirrors the required zip layout)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EMB = ROOT / "embeddings"
MODELS = ROOT / "models"
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
for _d in (DATA, EMB, MODELS, FIG, OUT):
    _d.mkdir(parents=True, exist_ok=True)

CLEANED_TXT = DATA / "cleaned.txt"
RAW_TXT = DATA / "raw.txt"
METADATA_JSON = DATA / "Metadata.json"

TFIDF_NPY = EMB / "tfidf_matrix.npy"
PPMI_NPY = EMB / "ppmi_matrix.npy"
W2V_NPY = EMB / "embeddings_w2v.npy"
WORD2IDX_JSON = EMB / "word2idx.json"

# ----------------------------------------------------------------------------
# Global
# ----------------------------------------------------------------------------
SEED = 42
VOCAB_SIZE = 10_000          # 10,000 most frequent tokens; the rest -> <UNK>
UNK = "<UNK>"
PAD = "<PAD>"
CLS = "<CLS>"
NUM = "<NUM>"                # number placeholder produced by the A1 tokenizer
PUNCT = set("۔؟!،؛:()\"'“”‘’«»-—%")

CATEGORIES = ["Politics", "Sports", "Economy", "International", "Health & Society"]

# ----------------------------------------------------------------------------
# Part 1 - embeddings
# ----------------------------------------------------------------------------
PPMI_WINDOW = 5              # symmetric context window k
W2V_DIM = 100                # d
W2V_WINDOW = 5               # k
W2V_NEG = 10                 # K noise samples per positive pair
W2V_LR = 1e-3                # Adam learning rate (eta)
W2V_BATCH = 512              # >= 512 required
W2V_EPOCHS = 5               # >= 5 required
W2V_NOISE_POWER = 0.75       # P_n(w) ~ f(w)^(3/4)
W2V_TABLE_SIZE = 10_000_000  # unigram noise table size (as in word2vec.c)
W2V_SUBSAMPLE = None         # frequent-word sub-sampling threshold t (None = off)

# ----------------------------------------------------------------------------
# Part 2 - sequence labelling
# ----------------------------------------------------------------------------
POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "CONJ", "POST",
            "NUM", "PUNC", "AUX", "UNK"]          # 11 listed in the brief + AUX
NER_TAGS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG",
            "B-MISC", "I-MISC"]
N_SENTENCES = 500
SENT_PER_CATEGORY = 100
MIN_SENT_LEN, MAX_SENT_LEN = 5, 50
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.5
SEQ_LR = 1e-3
SEQ_WEIGHT_DECAY = 1e-4
SEQ_BATCH = 16
SEQ_MAX_EPOCHS = 60
SEQ_PATIENCE = 5

# ----------------------------------------------------------------------------
# Part 3 - transformer
# ----------------------------------------------------------------------------
MAX_LEN = 256
D_MODEL = 128
N_HEADS = 4
D_K = D_V = 32
D_FF = 512
N_LAYERS = 4
TF_DROPOUT = 0.1
TF_LR = 5e-4
TF_WEIGHT_DECAY = 0.01
TF_WARMUP = 50
TF_EPOCHS = 20
TF_BATCH = 16
