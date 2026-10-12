"""Model 1 / Head C (docs/AGENT_PLAN.html §2): what is the subject? 90 animal species + human, "unknown" by threshold.

    python3 scripts/train_recognizer.py --push     # local: run this file as a Kaggle CPU kernel, download the result
    python3 train_recognizer.py                    # on Kaggle (what --push runs): extract features, fit, evaluate

Frozen DINOv2-small (LoRA off in the full Model 1), image -> [CLS, mean of patch tokens] = 768 numbers, then a
logistic-regression head. Data needs no labelling: Animals-90 (60 photos per species), Sea Animals (v2: a generic
"fish" class and ~20 sea species, at most CAP_SEA photos each) and LFW faces (class "human").
Evaluation:
  closed set  80/20 stratified split: top-1 / top-5 accuracy
  open set    10 species held out entirely. Three "unknown" scores are compared (v2): top softmax probability,
              cosine similarity to the nearest class prototype, and to the K-th nearest training image. For each,
              tau = 5th percentile on known test images; report AUROC and the share of held-out images below tau.
              The best score by AUROC is saved as the method the agent uses.
  in domain   the label-set subjects (pilot/labelset/*/candidates/S*.png, alpha over gray as camo.pilot normalizes)
Output (outputs/recognizer/): head.npz (W, b, mean, scale), meta.json (classes, prompt names, tau), report.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time

KAGGLE_USER = "dangkhoi3107"
KERNEL = f"{KAGGLE_USER}/camo-recognizer"
DATASETS = ["iamsouravbanerjee/animal-image-dataset-90-different-animals", "vencerlanz09/sea-animals-image-dataste",
            "jessicali9530/lfw-dataset", f"{KAGGLE_USER}/camo-code"]
CAP_SEA, K_NN = 120, 5
# Sea Animals folder (lower case) -> class name shared with Animals-90 where the species exists there
SEA_NAMES = {"jelly fish": "jellyfish", "sharks": "shark", "turtle_tortoise": "turtle", "crabs": "crab", "clams": "clam",
             "corals": "coral", "nudibranchs": "nudibranch", "puffers": "pufferfish", "sea rays": "ray",
             "sea urchins": "sea urchin", "whale": "whale", "seal": "seal", "dolphin": "dolphin"}
HUMAN = "human"
N_HUMAN = 600  # LFW has 13k faces; keep the human class about as large as 10 species
# species folder name -> word used in the prompt "a hidden {animal} silhouette ..."
PROMPT_NAME = {"goldfish": "fish", "pufferfish": "fish", "ladybugs": "ladybug", "pelecaniformes": "pelican",
               "sea urchin": "sea urchin", HUMAN: "human face, portrait"}
SEED = 0


# ---------------------------------------------------------------- local: push and fetch
def push() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kdir = os.path.join(root, "dist", "kernel_recognizer")
    shutil.rmtree(kdir, ignore_errors=True)
    os.makedirs(kdir)
    shutil.copy2(os.path.abspath(__file__), os.path.join(kdir, "train_recognizer.py"))
    with open(os.path.join(kdir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"id": KERNEL, "title": "camo-recognizer", "code_file": "train_recognizer.py", "language": "python",
                   "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True,
                   "dataset_sources": DATASETS, "competition_sources": [], "kernel_sources": []}, f, indent=1)
    subprocess.run(["kaggle", "kernels", "push", "-p", kdir], check=True)
    print(f"Running on Kaggle CPU: https://www.kaggle.com/code/{KERNEL}", flush=True)
    while True:
        time.sleep(60)
        out = subprocess.run(["kaggle", "kernels", "status", KERNEL], capture_output=True, text=True)
        status = (out.stdout + out.stderr).lower()
        print(time.strftime("%H:%M"), status.strip().splitlines()[-1] if status.strip() else "?", flush=True)
        if "complete" in status or "kernelworkerstatus.error" in status or "kernelworkerstatus.cancel" in status:
            break
    dest = os.path.join(root, "outputs", "recognizer")
    os.makedirs(dest, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", KERNEL, "-p", dest], check=False)
    if not os.path.exists(os.path.join(dest, "report.json")):
        sys.exit(f"No report.json: see the log in {dest} or https://www.kaggle.com/code/{KERNEL}")
    with open(os.path.join(dest, "report.json"), encoding="utf-8") as f:
        print(json.dumps(json.load(f)["summary"], indent=1, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------- Kaggle: data
def find_dirs(name: str) -> list:
    return sorted(os.path.join(r, name) for r, dirs, _ in os.walk("/kaggle/input") if name in dirs)


def animal_images() -> dict:
    """{species: [path, ...]} from the folder that holds both cat/ and lizard/."""
    for cat in find_dirs("cat"):
        root = os.path.dirname(cat)
        if os.path.isdir(os.path.join(root, "lizard")):
            out = {}
            for d in sorted(os.listdir(root)):
                files = sorted(glob.glob(os.path.join(root, d, "*.[jJpP][pPnN][gG]")))
                if files:
                    out[d.lower()] = files
            return out
    raise SystemExit("Animals-90 not found under /kaggle/input")


def sea_images(rng: random.Random) -> dict:
    """{class: [path, ...]} from the Sea Animals folder (the one that holds a Fish/ subfolder)."""
    for r, dirs, _ in os.walk("/kaggle/input"):
        if any(d.lower() == "fish" for d in dirs) and any(d.lower() in ("octopus", "seahorse") for d in dirs):
            out = {}
            for d in sorted(dirs):
                files = sorted(os.path.join(r, d, f) for f in os.listdir(os.path.join(r, d))
                               if f.lower().endswith((".jpg", ".jpeg", ".png")))
                if files:
                    name = SEA_NAMES.get(d.lower(), d.lower())
                    out[name] = rng.sample(files, min(CAP_SEA, len(files)))
            return out
    raise SystemExit("Sea Animals not found under /kaggle/input")


def face_images(rng: random.Random) -> list:
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk("/kaggle/input") if "lfw" in r.lower()
                   for f in fs if f.lower().endswith(".jpg"))
    if not files:
        raise SystemExit("LFW not found under /kaggle/input")
    by_person = {}
    for p in files:  # one face per person first, so the class is not dominated by a few famous people
        by_person.setdefault(os.path.basename(os.path.dirname(p)), []).append(p)
    people = sorted(by_person)
    rng.shuffle(people)
    return [by_person[p][0] for p in people[:N_HUMAN]]


def subject_images() -> list:
    """[(path, true class)] for the label-set subjects shipped in camo-code."""
    import yaml

    out = {}
    for path in glob.glob("/kaggle/input/**/pilot/labelset/b*/inputs.yaml", recursive=True):
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        for sid, meta in doc["subjects"].items():
            img = os.path.join(os.path.dirname(path), meta["path"])
            truth = HUMAN if meta["category"] == "human_portrait" else meta["label"]
            out[sid] = (img, truth)
    return [out[k] for k in sorted(out)]


# ---------------------------------------------------------------- Kaggle: features + head
def load_rgb(path: str):
    from PIL import Image

    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (128, 128, 128, 255))  # camo.pilot normalize: alpha over RGB(128)
        bg.alpha_composite(img)
        img = bg
    return img.convert("RGB")


def features(paths: list, batch: int = 32):
    import numpy as np
    import torch
    from transformers import AutoImageProcessor, AutoModel

    torch.set_num_threads(os.cpu_count() or 4)
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    out = []
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(paths), batch):
            imgs = [load_rgb(p) for p in paths[i:i + batch]]
            h = model(**proc(images=imgs, return_tensors="pt")).last_hidden_state
            out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).numpy())
            if i // batch % 20 == 0:
                print(f"features {i + len(imgs)}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
    return np.concatenate(out).astype("float32")


def fit(X, y, n_classes: int):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    mean, scale = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=0.5, max_iter=3000)
    clf.fit((X - mean) / scale, y)
    W = np.zeros((n_classes, X.shape[1]), "float32")
    b = np.full(n_classes, -1e9, "float32")
    W[clf.classes_], b[clf.classes_] = clf.coef_, clf.intercept_
    return {"W": W, "b": b, "mean": mean, "scale": scale}


def predict(head, X):
    import numpy as np

    z = ((X - head["mean"]) / head["scale"]) @ head["W"].T + head["b"]
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def unit(X, mu):
    """Centred, L2-normalised features: dot products are cosine similarities."""
    import numpy as np

    Z = X - mu
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)


def prototypes(B, y, n_classes: int):
    import numpy as np

    P = np.zeros((n_classes, B.shape[1]), "float32")
    for c in np.unique(y):
        m = B[y == c].mean(0)
        P[c] = m / (np.linalg.norm(m) + 1e-8)
    return P


def known_score(method: str, head, Xq):
    """Higher = looks like a class seen in training. head holds W/b (softmax) and mu/protos/bank (cosine scores)."""
    import numpy as np

    if method == "softmax":
        return predict(head, Xq).max(1)
    U = unit(Xq, head["mu"])
    if method == "prototype":
        return (U @ head["protos"].T).max(1)
    sims = U @ head["bank"].T.astype("float32")
    return -np.partition(-sims, K_NN - 1, axis=1)[:, K_NN - 1]  # K-th most similar training image


def build_head(X, y, n_classes: int):
    head = fit(X, y, n_classes)
    head["mu"] = X.mean(0)
    B = unit(X, head["mu"])
    head["protos"] = prototypes(B, y, n_classes)
    head["bank"] = B.astype("float16")
    head["bank_y"] = y
    return head


def train() -> int:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    rng = random.Random(SEED)
    animals = animal_images()
    for name, files in sea_images(rng).items():
        animals.setdefault(name, []).extend(files)
    faces = face_images(rng)
    classes = sorted(animals) + [HUMAN]
    paths, y = [], []
    for c, files in sorted(animals.items()):
        paths += files
        y += [classes.index(c)] * len(files)
    paths += faces
    y += [classes.index(HUMAN)] * len(faces)
    y = np.array(y)
    print(f"{len(classes)} classes, {len(paths)} images ({len(faces)} faces)", flush=True)
    subjects = subject_images()
    X = features(paths + [p for p, _ in subjects])
    Xs, X = X[len(paths):], X[:len(paths)]

    # closed set: 80/20 stratified
    idx = np.arange(len(y))
    test = np.array([i for c in range(len(classes)) for i in rng.sample(list(idx[y == c]), max(1, len(idx[y == c]) // 5))])
    train_ = np.setdiff1d(idx, test)
    head = fit(X[train_], y[train_], len(classes))
    P = predict(head, X[test])
    top1 = float((P.argmax(1) == y[test]).mean())
    top5 = float(np.mean([y[test][i] in np.argsort(-P[i])[:5] for i in range(len(test))]))
    per_class = {classes[c]: round(float((P.argmax(1)[y[test] == c] == c).mean()), 3) for c in range(len(classes))}

    # open set: hold out 10 species entirely, compare the three "unknown" scores
    held = set(rng.sample([c for c in classes if c != HUMAN], 10))
    known = np.array([classes[c] not in held for c in y])
    tr = train_[known[train_]]
    head_o = build_head(X[tr], y[tr], len(classes))
    kt, ut = test[known[test]], np.where(~known)[0]
    methods = {}
    for m in ("softmax", "prototype", "knn"):
        sk, su = known_score(m, head_o, X[kt]), known_score(m, head_o, X[ut])
        tau = float(np.percentile(sk, 5))
        auroc = roc_auc_score(np.r_[np.ones(len(sk)), np.zeros(len(su))], np.r_[sk, su])
        methods[m] = {"tau": round(tau, 4), "auroc": round(float(auroc), 3), "known_accepted": round(float((sk >= tau).mean()), 3),
                      "unknown_rejected": round(float((su < tau).mean()), 3)}
    best = max(methods, key=lambda m: methods[m]["auroc"])
    tau = methods[best]["tau"]

    # final head on everything; tau of the best method from the open-set run
    final = build_head(X, y, len(classes))
    Ps, Ss = predict(final, Xs), known_score(best, final, Xs)
    name = lambda c: PROMPT_NAME.get(c, c)  # noqa: E731
    subj = []
    for (path, truth), p, sc in zip(subjects, Ps, Ss):
        k = int(p.argmax())
        subj.append({"subject": os.path.basename(path), "truth": truth, "pred": classes[k], "prompt": name(classes[k]),
                     "prob": round(float(p[k]), 3), "score": round(float(sc), 3), "unknown": bool(sc < tau),
                     "top3": [classes[j] for j in np.argsort(-p)[:3]]})
    ok = [s for s in subj if s["prompt"] == name(s["truth"])]
    summary = {"classes": len(classes), "images": len(paths), "closed_top1": round(top1, 3), "closed_top5": round(top5, 3),
               "unknown_method": best, **{k: v for k, v in methods[best].items()},
               "subjects_correct": f"{len(ok)}/{len(subj)}",
               "subjects_marked_unknown": sum(s["unknown"] for s in subj)}
    out = "/kaggle/working"
    keep = {k: v for k, v in final.items() if best == "knn" or k not in ("bank", "bank_y")}
    np.savez(os.path.join(out, "head.npz"), **keep)
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"version": 2, "backbone": "facebook/dinov2-small", "feature": "concat(CLS, mean(patch tokens))",
                   "classes": classes, "prompt_name": {c: name(c) for c in classes},
                   "category": {c: "human_portrait" if c == HUMAN else "animal" for c in classes},
                   "unknown_method": best, "k_nn": K_NN, "tau": tau}, f, indent=1)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "held_out_species": sorted(held), "unknown_methods": methods,
                   "per_class_top1": per_class, "subjects": subj}, f, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--push", action="store_true", help="run on Kaggle and download outputs/recognizer/")
    a, _ = ap.parse_known_args()
    sys.exit(push() if a.push else train())
