"""Build ImageNet-100 (the 100-class CMC split of Tian et al., imagenet100_classes.txt)
in the layout train.py --dataset imagenet100 expects: <out>/{train,val}/<wnid>/*.JPEG.

Three sources:
  --source kaggle --src <dir>                    (Kaggle `ambityga/imagenet100`: train.X1..X4/<wnid>/,
                                                 val.X/<wnid>/, Labels.json. NOTE this is NOT the CMC
                                                 split: 100 other classes, 1,300 train images each =
                                                 130,000 train / 5,000 val)
  --source imagenet1k --src /data/ILSVRC2012   (ImageFolder layout: train/<wnid>/, val/<wnid>/;
                                                 run the standard valprep script first if
                                                 val/ is still flat)
  --source hf                                    (Hugging Face `clane9/imagenet-100`, same
                                                 CMC classes; needs `pip install datasets`)

Images are re-encoded with the shorter side resized to --short_side (default 256,
quality 90): RandomResizedCrop(224) / Resize(256)+CenterCrop(224) then see the same
pixels as from the originals, but JPEG decoding -- the data-loading bottleneck on a
single GPU when every instance is forwarded every epoch -- gets ~2x cheaper.
"""
import argparse
import os
from multiprocessing import Pool

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def resize_save(job):
    src, dst, short = job
    if os.path.exists(dst):
        return 0
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            if short > 0 and min(w, h) > short:
                s = short / min(w, h)
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
            im.save(dst, "JPEG", quality=90)
        return 1
    except Exception as e:  # report and keep going; a missing file is caught by the count check
        print(f"FAILED {src}: {e}")
        return 0


def from_imagenet1k(a, wnids):
    jobs = []
    for split in ("train", "val"):
        for wnid in wnids:
            sdir = os.path.join(a.src, split, wnid)
            if not os.path.isdir(sdir):
                raise SystemExit(f"missing {sdir} (val/ must be in ImageFolder layout)")
            ddir = os.path.join(a.out, split, wnid)
            os.makedirs(ddir, exist_ok=True)
            for fn in os.listdir(sdir):
                base = os.path.splitext(fn)[0]
                jobs.append((os.path.join(sdir, fn), os.path.join(ddir, base + ".JPEG"), a.short_side))
    with Pool(a.workers) as p:
        done = sum(p.imap_unordered(resize_save, jobs, chunksize=64))
    print(f"wrote {done} new images ({len(jobs)} total)")


def from_kaggle(a):
    jobs, wnids = [], set()
    dirs = {"train": sorted(d for d in os.listdir(a.src) if d.startswith("train.X")), "val": ["val.X"]}
    for split, subdirs in dirs.items():
        for sub in subdirs:
            base = os.path.join(a.src, sub)
            if not os.path.isdir(base):
                raise SystemExit(f"missing {base}")
            for wnid in sorted(os.listdir(base)):
                sdir = os.path.join(base, wnid)
                if not os.path.isdir(sdir):
                    continue
                if split == "train":
                    wnids.add(wnid)
                ddir = os.path.join(a.out, split, wnid)
                os.makedirs(ddir, exist_ok=True)
                for fn in os.listdir(sdir):
                    base_fn = os.path.splitext(fn)[0]
                    jobs.append((os.path.join(sdir, fn), os.path.join(ddir, base_fn + ".JPEG"), a.short_side))
    if len(wnids) != 100:
        raise SystemExit(f"expected 100 classes under {a.src}, found {len(wnids)}")
    with Pool(a.workers) as p:
        done = sum(p.imap_unordered(resize_save, jobs, chunksize=64))
    print(f"wrote {done} new images ({len(jobs)} total)")


def from_hf(a, wnids):
    from datasets import load_dataset
    ds = load_dataset("clane9/imagenet-100")
    for split_hf, split in (("train", "train"), ("validation", "val")):
        d = ds[split_hf]
        names = d.features["label"].names
        missing = set(wnids) - set(n.split(",")[0].strip() for n in names)
        if missing and len(set(names) & set(wnids)) < 100:
            raise SystemExit(f"HF label names do not match imagenet100_classes.txt; first labels: {names[:5]}")
        for i, ex in enumerate(d):
            wnid = names[ex["label"]].split(",")[0].strip()
            ddir = os.path.join(a.out, split, wnid)
            os.makedirs(ddir, exist_ok=True)
            dst = os.path.join(ddir, f"{split}_{i:07d}.JPEG")
            if os.path.exists(dst):
                continue
            im = ex["image"].convert("RGB")
            w, h = im.size
            if a.short_side > 0 and min(w, h) > a.short_side:
                s = a.short_side / min(w, h)
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
            im.save(dst, "JPEG", quality=90)
        print(f"{split}: {len(d)} images")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["imagenet1k", "hf", "kaggle"], required=True)
    p.add_argument("--src", default=None, help="ImageNet-1k root (imagenet1k source)")
    p.add_argument("--out", default="./imagenet100")
    p.add_argument("--short_side", type=int, default=256, help="0 keeps original resolution")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    wnids = []
    if a.source != "kaggle":  # kaggle variant has its own (non-CMC) 100 classes
        with open(os.path.join(HERE, "imagenet100_classes.txt")) as f:
            wnids = [w.strip() for w in f if w.strip()]
        assert len(wnids) == 100
    if a.source == "kaggle":
        if not a.src:
            raise SystemExit("--src is required for --source kaggle")
        from_kaggle(a)
    elif a.source == "imagenet1k":
        if not a.src:
            raise SystemExit("--src is required for --source imagenet1k")
        from_imagenet1k(a, wnids)
    else:
        from_hf(a, wnids)
    for split in ("train", "val"):
        n = sum(len(os.listdir(os.path.join(a.out, split, w))) for w in sorted(os.listdir(os.path.join(a.out, split))))
        print(f"{split}: {len(os.listdir(os.path.join(a.out, split)))} classes, {n} images "
              f"(expected {'130,000' if a.source == 'kaggle' else '~126,689'} train / 5,000 val)")


if __name__ == "__main__":
    main()
