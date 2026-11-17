"""
The inference tool: image path in -> APPROVE / HUMAN_REVIEW / REJECT out.

Loads the model bundle saved from the Colab notebook (section 10) and applies
the thresholds chosen there. The agent calls infer(); you can also run it alone:

    python fraud_tool.py path/to/document.jpg
"""
import json
import os
import sys

import cv2
import joblib
import numpy as np

from fraud_features import extract_features

BUNDLE_PATH = os.environ.get("FRAUD_MODEL", "fraud_model_v2.joblib")
_bundle = None


def load_bundle(path=BUNDLE_PATH):
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(path)
    return _bundle


# STEP 6 (DJ's flow): probability of fake -> business decision.
# Thresholds come from notebook section 9.0 (at most 1% of genuine docs auto-rejected).
def decide(p_fake, t_approve, t_reject):
    if p_fake > t_reject:
        return "REJECT"
    if p_fake < t_approve:
        return "APPROVE"
    return "HUMAN_REVIEW"


# STEP 5a (DJ's flow): extract the same 295 features the model was trained on.
# The GUI calls this BEFORE the agent (DJ's synchronous flow); infer() calls it for CLI use.
def features_from_path(path):
    """Return (features, error). features is a 1 x 295 array, or None if the file can't be used."""
    if not os.path.isfile(path):
        return None, "File not found."
    if cv2.imread(path) is None:
        return None, "Could not read the image. Upload a clear JPG or PNG."
    return np.nan_to_num(extract_features(path)[0].reshape(1, -1)), None  # same cleanup as the notebook


# STEP 5b + 6 (DJ's flow): invoke the team's trained classifier on extracted features,
# then turn the probability into a decision.
def classify_features(feats, name="document"):
    b = load_bundle()   # the model + thresholds saved by notebook section 9.1
    if feats.shape[1] != b["n_features"]:
        return {"file": name, "error": f"Expected {b['n_features']} features, got {feats.shape[1]}. "
                                       "fraud_features.py does not match the notebook."}
    p_fake = float(b["model"].predict_proba(feats)[0, 1])
    return {
        "file": name,
        "decision": decide(p_fake, b["t_approve"], b["t_reject"]),
        "probability_fake": round(p_fake, 3),
        "thresholds": {"approve_below": round(b["t_approve"], 3), "reject_above": round(b["t_reject"], 3)},
        "model": b["model_name"],
    }


# STEP 5 (DJ's flow): the full "infer" method, image path in -> decision out.
# DJ: "take the path of that particular file, extract the necessary features,
#      and then invoke ... your model". Same logic as notebook section 9.2.
def infer(path):
    """Return a dict with the decision and probability of fake for one document image."""
    feats, err = features_from_path(path)
    if err:
        return {"file": path, "error": err}
    return classify_features(feats, os.path.basename(path))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python fraud_tool.py path/to/document.jpg")
    for p in sys.argv[1:]:
        print(json.dumps(infer(p), indent=2))
