# Detecting Phishing and Spam in Student Email

A text classifier that reads an email (subject + body) and labels it **malicious** (phishing or spam)
or **legitimate**. Course project for GE2340 *AI: Past, Present & Future* (City University of Hong Kong), Phase 2.

Everything lives in one notebook: [`phishing_classifier.ipynb`](phishing_classifier.ipynb).
GitHub shows it **already executed**, so you can read every table and figure without running anything.

## Main results (test set, 16,416 emails)

| Model | Precision | Recall | F1 | Missed malicious (FN) |
|---|---|---|---|---|
| Baseline (always "malicious") | 0.522 | 1.000 | 0.686 | 0 |
| **A: Logistic regression** (threshold 0.297) | 0.983 | 0.994 | 0.989 | 49 |
| B: Neural network (MLP, threshold 0.241) | 0.984 | 0.994 | 0.989 | 49 |

- A and B are **not significantly different** (McNemar exact test, p = 0.29). B has 64× more parameters
  and takes about 7× longer to train (laptop CPU; exact seconds vary between runs).
- **Caveat:** the model partly learns the *style of each source corpus*. When a corpus is left out of
  training, performance on it drops (e.g. recall on unseen Nazario phishing falls from 0.99 to 0.59;
  0.51–0.68 across variants of the corpus-matching step). See section 11 of the notebook.

## Quick path

### Option 1: Google Colab

1. Open `phishing_classifier.ipynb` in [Google Colab](https://colab.research.google.com/) (*File → Upload notebook*).
2. Download the dataset from Kaggle (next section) and upload the CSV files into a folder called `data/`
   in the Colab file panel.
3. *Runtime → Run all*. A full run takes about 15 minutes (most of it is the neural network).

### Option 2: Locally

```bash
git clone https://github.com/victorr-cm/phishing-email-classifier.git
cd phishing-email-classifier
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# put the CSV files in data/ (next section), then:
jupyter notebook phishing_classifier.ipynb
```

## Live demo: classify a new email

`demo.py` loads the saved final model (`models/model_a.joblib`, written by section 5.4 of the notebook)
and classifies an email in about a second. It does **not** train anything and does **not** need the dataset.

```bash
python demo.py demo_emails/a_phishing.txt      # one email
python demo.py demo_emails/                    # every .txt file in the folder
```

For each email it prints the email text, the probability of being malicious, the decision at the threshold chosen on
validation (0.297), and the 3 terms that push the decision most towards each side (θⱼ · xⱼ):

```
 P(malicious) = 0.250     Decision: SAFE  (threshold 0.297)
 Pushing towards MALICIOUS        Pushing towards SAFE
   suspended            +1.37     university           -7.37
```

- `demo_emails/` contains four example emails: a classic phishing email, a legitimate email from a lecturer,
  a phishing email imitating CityU IT Services, and the same email with "university" replaced by "bank".
- Email files are plain text; a first line starting with `Subject:` is used as the subject.
- New emails are pre-processed like the dataset (lower-case, punctuation deleted, NLTK English stop words removed).
  This is our reconstruction of the dataset authors' pre-processing, checked against the data, not their exact code.
- The saved model loads with scikit-learn 1.9.x (the version used to train it).

## Data

Dataset: [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
(Al-Subaiey et al., 2024, [arXiv:2405.11619](https://arxiv.org/abs/2405.11619)). Download it from Kaggle
and place the files like this:

```
data/
├── phishing_email.csv     ← main dataset (82,486 emails), used by sections 1–10
├── CEAS_08.csv            ┐
├── Enron.csv              │
├── Ling.csv               │  per-corpus files, only needed for section 11
├── Nazario.csv            │  (recovering which corpus each email comes from)
├── Nigerian_Fraud.csv     │
└── SpamAssasin.csv        ┘  (the misspelling is Kaggle's file name)
```

`data/` is in `.gitignore`: the dataset is not redistributed in this repository.

## What the notebook does

| Step | Decision |
|---|---|
| Cleaning | Drop empty and duplicated emails **before** splitting: 82,486 → 82,077 emails |
| Split | Stratified 60 / 20 / 20 (train / validation / test), `random_state=42` |
| Features | TF-IDF, unigrams + bigrams, 15,000 terms, `min_df=5`, fitted on training data only |
| Baseline | `DummyClassifier`, always predicts the majority class |
| Model A | `LogisticRegression`, L2. `C` ∈ {0.01, 0.1, 1, 10, 100, 1000} chosen by validation F1 → **C = 100** |
| Threshold | Chosen on validation by maximising F2 (recall weighted 2× precision) |
| Final fit | Whole pipeline refitted on train + validation, evaluated **once** on test |
| Model B | `MLPClassifier`: 1 hidden layer of 64 ReLU units, Adam, early stopping. Same features and protocol as A |
| Analysis | Confusion matrix, precision-recall curve, top-20 weights, false negatives, per-corpus results, leave-one-corpus-out |

## Repository layout

```
phishing_classifier.ipynb   the whole pipeline (executed, with outputs)
demo.py                     classifies new emails with the saved model and explains the decision
models/model_a.joblib       final logistic regression model (saved by notebook section 5.4)
demo_emails/                example emails for the demo
figures/                    report figures (PDF for LaTeX, PNG for viewing)
results/                    result tables as CSV (test results, top weights, false negatives, per-corpus, LOCO)
requirements.txt
```

## Reproducibility notes

- All random steps use `random_state=42`, so the metrics are reproducible.
- Training times in the notebook were measured on a laptop CPU. On Colab the absolute seconds differ,
  but the ratio between models stays similar.

## Authors

Victor Carilla, Marta Bach, Nil Gallart (City University of Hong Kong).
