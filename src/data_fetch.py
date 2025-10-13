import os, json, random, csv
from pathlib import Path
from typing import List, Tuple

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def _ensure_sms_csv() -> Path:
    """Create a small vendored SMS dataset if not present.
    Format: label,text  (labels: ham|spam)
    """
    csv_path = DATA_DIR / "sms_spam.csv"
    if csv_path.exists():
        return csv_path
    samples = [
        ("spam","WINNER!! As a valued network customer you have been selected to receive a prize. Call now!"),
        ("spam","URGENT! You have won a 1 week FREE membership. Reply WIN to claim."),
        ("spam","Congratulations! You have been selected for a $1000 Walmart gift card. Click link."),
        ("ham","Hey, can we move the meeting to 3pm?"),
        ("ham","I’ll be there in 5 minutes."),
        ("ham","Don’t forget to bring the documents."),
        ("ham","Dinner tonight? I can cook."),
        ("spam","You are approved for a low interest loan. Reply YES to proceed."),
        ("ham","Happy birthday! Hope you have a great day."),
        ("spam","FREE entry in 2 a weekly competition to win FA Cup tickets. Text WIN to 12345 now!")
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label","text"])
        w.writerows(samples)
    return csv_path

def _split(rows: List[Tuple[str,str]], seed=1337, version="v1"):
    random.Random(seed).shuffle(rows)
    # Slight variation for v2 (simulate drift / class balance change)
    if version == "v2":
        rows = rows + [("spam","Claim your free prize now")] * 4  # bump spam rate a bit
    n = len(rows)
    n_train = int(0.7*n)
    n_valid = int(0.15*n)
    train = rows[:n_train]
    valid = rows[n_train:n_train+n_valid]
    test  = rows[n_train+n_valid:]
    return train, valid, test

def fetch():
    source = os.getenv("DATA_SOURCE","sms_spam")
    version = os.getenv("DATA_VERSION","v1")
    assert source in {"sms_spam","synthetic"}
    if source == "sms_spam":
        path = _ensure_sms_csv()
        with path.open(encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            rows = [(r["label"], r["text"]) for r in rdr]
    else:
        # simple synthetic generator
        rows = []
        for i in range(400):
            rows.append(("ham", f"note {i} meeting at {i%12+1}pm"))
        for i in range(100):
            rows.append(("spam","win cash now free limited time offer"))
    train, valid, test = _split(rows, version=version)

    for name, subset in [("train",train),("valid",valid),("test",test)]:
        out = DATA_DIR / f"{name}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text","label"])
            w.writerows([(t,l) for (l,t) in subset])
    print("Wrote data/train.csv, data/valid.csv, data/test.csv")

if __name__ == "__main__":
    fetch()
