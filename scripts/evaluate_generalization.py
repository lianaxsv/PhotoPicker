"""Held-out robustness benchmark for PhotoPick overexposure classification.
Independent of the original five image calibration set. Run from project root:
python scripts/evaluate_generalization.py --output results/heldout.csv
Requires scikit-image. Synthetic brightening is not a human exposure label.
"""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
from skimage import data
from photo_quality.core import analyze_image

NAMES = ("hubble_deep_field","grass","gravel","moon","page","text","brick","coins","horse","immunohistochemistry","retina")
SIZES = (128, 256, 512)

def run():
    rows=[]
    for name in NAMES:
        raw=getattr(data,name)()
        if raw.dtype == np.bool_:
            raw=raw.astype("uint8")*255
        if raw.dtype != np.uint8:
            raw=np.clip(raw,0,255).astype("uint8")
        bgr=cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR if raw.ndim==2 else cv2.COLOR_RGB2BGR)
        for size in SIZES:
            ref=cv2.resize(bgr,(size,size),interpolation=cv2.INTER_AREA)
            variants={
                "reference":ref,
                "moderate_brightening":np.clip(ref.astype("float32")*1.3+10,0,255).astype("uint8"),
                "severe_brightening":np.clip(ref.astype("float32")*1.7+35,0,255).astype("uint8"),
                "extreme_brightening":np.clip(ref.astype("float32")*2.0+60,0,255).astype("uint8"),
                "contrast_reduction":np.clip(ref.astype("float32")*.35+82,0,255).astype("uint8"),
            }
            for condition,img in variants.items():
                m=analyze_image(img)
                rows.append(dict(image=name,size=size,condition=condition,
                    brightness=m["brightness"],clipped_fraction=m["highlight_fraction"],
                    before=int(m["brightness"]>205),
                    after=int("overexposed" in m["issues"])))
    return rows

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results/heldout.csv")
    args=parser.parse_args()
    rows=run()
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    for condition in sorted(set(x["condition"] for x in rows)):
        r=[x for x in rows if x["condition"]==condition]
        print(f"{condition}: before={sum(x['before'] for x in r)}/{len(r)}, after={sum(x['after'] for x in r)}/{len(r)}")
    print("Full results:",out)
