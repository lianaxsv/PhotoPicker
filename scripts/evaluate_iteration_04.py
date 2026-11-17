"""Iteration 4 experiment: audit an enhancement-skip candidate.

Run: pip install scikit-image; python scripts/evaluate_iteration_04.py
This script deliberately does NOT change production enhancement logic.
Synthetic reconstructions are not human image-quality judgments.
"""
import csv
from pathlib import Path
import cv2
import numpy as np
from skimage import data
from photo_quality.core import analyze_image, enhance_image, adaptive_gamma

DEVELOPMENT = ("astronaut","coffee","chelsea","camera","rocket")
SECONDARY = ("hubble_deep_field","grass","gravel","moon","page","text","brick","coins",
             "horse","immunohistochemistry","retina")

def load(name):
    img=getattr(data,name)()
    if img.dtype!=np.uint8:
        img=np.uint8(np.clip(img*255 if img.max()<=1 else img,0,255))
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR if img.ndim==2 else cv2.COLOR_RGB2BGR)
    return cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)

def mae(a,b):
    return float(np.mean(np.abs(a.astype("float32")-b.astype("float32"))))

def candidate(img):
    metrics=analyze_image(img)
    if metrics["brightness"]<75 and "underexposed" in metrics["issues"]:
        return adaptive_gamma(img)[0]
    return img.copy()

def evaluate():
    rows=[]
    for group,names in (("development",DEVELOPMENT),("secondary",SECONDARY)):
        for name in names:
            ref=load(name)
            variants={"clean":ref}
            variants.update({f"dark{int(s*100)}":(ref.astype("float32")*s).astype("uint8")
                             for s in (.25,.4,.55,.7,.85)})
            variants.update({
                "blur2":cv2.GaussianBlur(ref,(0,0),2),
                "blur5":cv2.GaussianBlur(ref,(0,0),5),
                "bright":np.uint8(np.clip(ref.astype("float32")*1.7+35,0,255)),
                "contrast":np.uint8(np.clip((ref.astype("float32")-127)*.2+127,0,255)),
            })
            rng=np.random.default_rng(sum(name.encode()))
            variants["noise18"]=np.uint8(np.clip(ref.astype("float32")+
                rng.normal(0,18,ref.shape),0,255))
            for defect,img in variants.items():
                current,_=enhance_image(img)
                proposal=candidate(img)
                rows.append(dict(group=group,image=name,defect=defect,
                    mae_input=round(mae(ref,img),5),
                    mae_current=round(mae(ref,current),5),
                    mae_candidate=round(mae(ref,proposal),5)))
    return rows

if __name__=="__main__":
    rows=evaluate()
    path=Path("results/iteration_04.csv")
    path.parent.mkdir(exist_ok=True)
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    for group in ("development","secondary"):
        section=[r for r in rows if r["group"]==group]
        for field in ("mae_input","mae_current","mae_candidate"):
            print(group,field,round(np.mean([r[field] for r in section]),2))
        print("worse vs input",sum(r["mae_current"]>r["mae_input"]+.1 for r in section),
              sum(r["mae_candidate"]>r["mae_input"]+.1 for r in section))
    print("Wrote",path)
