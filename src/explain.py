from typing import List, Tuple
import numpy as np
from sklearn.pipeline import Pipeline

def top_tokens_for_text(pipe: Pipeline, text: str, k: int = 3) -> List[Tuple[str, float]]:
    """Return top contributing tokens (by |coef * tfidf|) for a single text.
    Works for binary LR with tfidf + clf.
    """
    tfidf = pipe.named_steps["tfidf"]
    clf   = pipe.named_steps["clf"]
    feat_names = tfidf.get_feature_names_out()
    X = tfidf.transform([text])  # 1 x V
    coefs = clf.coef_.ravel()    # V
    contrib = X.toarray().ravel() * coefs
    idx = np.argsort(-np.abs(contrib))[:k]
    return [(feat_names[i], float(contrib[i])) for i in idx]
