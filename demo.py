"""Classify an email with the saved model A and explain the decision.

Usage:
    python demo.py demo_emails/a_phishing.txt
    python demo.py demo_emails/            (every .txt file in the folder)

The model is loaded from models/model_a.joblib, saved by section 5.4 of
phishing_classifier.ipynb. Nothing is trained here.

Email files are plain text. If the first line starts with "Subject:", it is
used as the subject and the rest as the body.
"""
import re
import sys
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).parent / "models" / "model_a.joblib"
TOP_K = 3
WIDTH = 86

# The dataset text was pre-processed by its authors: lower-cased, punctuation
# deleted and English stop words removed (we checked that they match NLTK's
# list: e.g. "the" survives in 0.3% of the emails, while "please", which is
# not in that list, appears in 29%). New emails get the same treatment, so the
# model sees text in the same form as during training.
STOP_WORDS = set("""
i me my myself we our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their theirs
themselves what which who whom this that these those am is are was were be
been being have has had having do does did doing a an the and but if or
because as until while of at by for with about against between into through
during before after above below to from up down in out on off over under again
further then once here there when where why how all any both each few more
most other some such no nor not only own same so than too very s t can will
just don should now d ll m o re ve y ain aren couldn didn doesn hadn hasn
haven isn ma mightn mustn needn shan shouldn wasn weren won wouldn
""".split())


def preprocess(text):
    text = re.sub(r"[^\w\s]", "", text.lower())   # delete punctuation, as in the dataset
    return " ".join(w for w in text.split() if w not in STOP_WORDS)


def read_email(path):
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    subject = ""
    if lines and lines[0].lower().startswith("subject:"):
        subject, lines = lines[0].split(":", 1)[1].strip(), lines[1:]
    return subject, "\n".join(lines).strip()


def explain(pipeline, threshold, path):
    subject, body = read_email(path)
    text = preprocess(subject + " " + body)
    tfidf, lr = pipeline.named_steps["tfidf"], pipeline.named_steps["lr"]

    x = tfidf.transform([text])
    p = pipeline.predict_proba([text])[0, 1]
    decision = "MALICIOUS" if p >= threshold else "SAFE"

    # Contribution of each term present in the email: theta_j * x_j.
    terms = tfidf.get_feature_names_out()
    contrib = sorted(((lr.coef_[0, j] * v, terms[j]) for j, v in zip(x.indices, x.data)), reverse=True)
    towards_mal = [c for c in contrib if c[0] > 0][:TOP_K]
    towards_safe = sorted(c for c in contrib if c[0] < 0)[:TOP_K]

    print("=" * WIDTH)
    print(f" Email:   {Path(path).name}")
    print(f" Subject: {subject}")
    print("-" * WIDTH)
    for line in body.splitlines():          # the email exactly as written (before pre-processing)
        print(f"   {line}")
    print("-" * WIDTH)
    print(f" P(malicious) = {p:.3f}     Decision: {decision}  (threshold {threshold:.3f})")
    print("-" * WIDTH)
    print(f" {'Pushing towards MALICIOUS':<32} Pushing towards SAFE")
    for i in range(max(len(towards_mal), len(towards_safe), 1)):
        left = f"{towards_mal[i][1]:<20} {towards_mal[i][0]:+.2f}" if i < len(towards_mal) else ""
        right = f"{towards_safe[i][1]:<20} {towards_safe[i][0]:+.2f}" if i < len(towards_safe) else ""
        print(f"   {left:<30} {right}")
    print("=" * WIDTH + "\n")


def main(args):
    if not args:
        sys.exit(__doc__)
    paths = []
    for arg in args:
        p = Path(arg)
        paths += sorted(p.glob("*.txt")) if p.is_dir() else [p]
    model = joblib.load(MODEL_PATH)
    for path in paths:
        explain(model["pipeline"], model["threshold"], path)


if __name__ == "__main__":
    main(sys.argv[1:])
