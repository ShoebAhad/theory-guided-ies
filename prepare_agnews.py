"""Tokenize AG News once into ./agnews/agnews_L64.npz (token ids padded with 0)."""
import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

L = 64
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agnews")
os.makedirs(out, exist_ok=True)
ds = load_dataset("fancyzhx/ag_news")
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
assert tok.pad_token_id == 0
arrs = {}
for split in ("train", "test"):
    enc = tok(list(ds[split]["text"]), truncation=True, max_length=L, padding="max_length")
    arrs[f"x_{split}"] = np.asarray(enc["input_ids"], dtype=np.int32)
    arrs[f"y_{split}"] = np.asarray(ds[split]["label"], dtype=np.int64)
    lens = (arrs[f"x_{split}"] != 0).sum(1)
    print(split, arrs[f"x_{split}"].shape, "classes", np.bincount(arrs[f"y_{split}"]), "mean len %.1f" % lens.mean(),
          "truncated %.1f%%" % (100 * (lens == L).mean()))
np.savez_compressed(os.path.join(out, f"agnews_L{L}.npz"), vocab_size=tok.vocab_size, **arrs)
print("vocab", tok.vocab_size)
