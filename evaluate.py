"""Evaluates the car make classifier on two splits.

A random split lets near identical photos of one car model land in both train and test. The
harder split holds out whole models, so the network must recognize a make from models it has
never seen. Comparing them shows how much of the random split score is memorization.
Usage: python evaluate.py   (reads archive/, writes results/)
"""
import json
import pathlib

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.nn import functional as F

SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load():
    xs, makes, models = [], [], []
    for path in sorted(pathlib.Path("archive").glob("*.jpg")):
        parts = path.name.split("_")
        try:
            image = Image.open(path).convert("RGB").resize((SIZE, SIZE))
        except OSError:
            continue
        xs.append(np.asarray(image, dtype=np.uint8))
        makes.append(parts[0])
        models.append(f"{parts[0]}_{parts[1]}")
    names, y = np.unique(makes, return_inverse=True)
    return np.stack(xs).transpose(0, 3, 1, 2).copy(), y, np.array(models), names


class CNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        layers, channels = [], 3
        for out in (32, 64, 128, 256):
            layers += [nn.Conv2d(channels, out, 3, padding=1), nn.BatchNorm2d(out), nn.ReLU(),
                       nn.Conv2d(out, out, 3, padding=1), nn.BatchNorm2d(out), nn.ReLU(), nn.MaxPool2d(2)]
            channels = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, classes))

    def forward(self, x):
        return self.head(self.features(x).flatten(2).mean(2))


def to_float(x):
    return (x.float() / 255 - 0.5) / 0.25


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(to_float(torch.from_numpy(x[i:i + 256]).to(DEVICE))).argmax(1).cpu().numpy()
                               for i in range(0, len(x), 256)])


def train(x, y, classes, epochs=20, patience=5, seed=0):
    torch.manual_seed(seed)
    x_fit, x_val, y_fit, y_val = train_test_split(x, y, test_size=0.1, stratify=y, random_state=seed)
    model = CNN(classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    weight = torch.tensor(np.sqrt(len(y_fit) / (classes * np.maximum(np.bincount(y_fit, minlength=classes), 1))),
                          dtype=torch.float32, device=DEVICE)
    rng = np.random.default_rng(seed)
    best, state, wait = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(x_fit))
        for start in range(0, len(order), 128):
            batch = order[start:start + 128]
            images = x_fit[batch]
            if rng.random() < 0.5:
                images = images[..., ::-1].copy()
            xb = to_float(torch.from_numpy(images).to(DEVICE))
            loss = F.cross_entropy(model(xb), torch.from_numpy(y_fit[batch]).to(DEVICE), weight=weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val = f1_score(y_val, predict(model, x_val), average="macro")
        print(f"epoch {epoch} val macro F1 {val:.4f}", flush=True)
        if val > best:
            best, wait = val, 0
            state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    return model


def score(y, p):
    return dict(accuracy=accuracy_score(y, p), balanced_accuracy=balanced_accuracy_score(y, p),
                macro_f1=f1_score(y, p, average="macro"))


def main():
    x, y, models, names = load()
    classes = len(names)
    largest = np.bincount(y).max() / len(y)
    idx = np.arange(len(y))
    random_tr, random_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
    group_tr, group_te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(x, y, models))
    results = dict(images=len(y), makes=classes, car_models=len(set(models)), largest_class_share=float(largest))
    for name, tr, te in (("random_split", random_tr, random_te), ("unseen_models_split", group_tr, group_te)):
        model = train(x[tr], y[tr], classes)
        results[name] = score(y[te], predict(model, x[te]))
        results[name]["test_images"] = len(te)
        print(name, results[name], flush=True)
    out = pathlib.Path("results")
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
