"""Compare pre-iteration scoring to current noise-aware scoring.

Reproduces 23 reference images x 3 Gaussian noise levels = 69 comparisons.
Requires scikit-image; run from project root:
python scripts/evaluate_ranking_iteration_03.py
Synthetic noise is not the same as human-judged photographic preference.
"""
import csv
from pathlib import Path
import cv2
import numpy as np
from skimage import data
from photo_quality.core import analyze_image

SETS = {
  "development": ("astronaut","coffee","chelsea","camera","rocket"),
  "previous_exposure": ("hubble_deep_field","grass","gravel","moon","page","text","brick","coins","horse","immunohistochemistry","retina"),
  "additional_content": ("clock","checkerboard","colorwheel","logo","microaneurysms","shepp_logan_phantom","lfw_subset"),
}

def load(name):
    img = getattr(data,name)()
    if name == "lfw_subset":
        img=img[0]
    if img.ndim==3 and img.shape[2]==4:
        img=img[:,:,:3]
    if img.dtype==np.bool_:
        img=img.astype(np.uint8)*255
    elif img.dtype!=np.uint8:
        img=np.clip(img*255 if img.max()<=1 else img,0,255).astype(np.uint8)
    img=cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img,cv2.COLOR_GRAY2BGR if img.ndim==2 else cv2.COLOR_RGB2BGR)

def baseline_score(bgr):
    # Exact old formula from the branch before Iteration 3.
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    brightness=float(gray.mean())
    contrast=float(gray.std())
    shadow=float(np.mean(gray<=12))
    highlight=float(np.mean(gray>=243))
    return round(100*float(np.clip(
        0.4*np.clip(np.log1p(sharp)/np.log1p(500),0,1)
       +0.35*np.clip(1-abs(brightness-128)/128,0,1)
       +0.25*np.clip(contrast/65,0,1)
       -np.clip((shadow+highlight)*0.5,0,0.4),0,1)),2)

def measure():
    rng=np.random.default_rng(5810)
    rows=[]
    for cohort,names in SETS.items():
        for name in names:
            source=load(name)
            old_original=baseline_score(source)
            new_original=analyze_image(source)["quality_score"]
            for sigma in (8,16,28):
                noisy=np.uint8(np.clip(source.astype(np.float32)
                    +rng.normal(0,sigma,source.shape),0,255))
                old_noisy=baseline_score(noisy)
                new_noisy=analyze_image(noisy)["quality_score"]
                rows.append(dict(cohort=cohort,image=name,sigma=sigma,
                    original_score_before=old_original,noisy_score_before=old_noisy,
                    noisy_wins_before=int(old_noisy>old_original),
                    original_score_after=new_original,noisy_score_after=new_noisy,
                    noisy_wins_after=int(new_noisy>new_original)))
    return rows

if __name__=="__main__":
    rows=measure()
    out=Path("results/ranking_iteration_03.csv")
    out.parent.mkdir(exist_ok=True)
    with out.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    for cohort in SETS:
        subset=[r for r in rows if r["cohort"]==cohort]
        print(f"{cohort}: noisy outranks original {sum(r['noisy_wins_before'] for r in subset)}/{len(subset)} -> {sum(r['noisy_wins_after'] for r in subset)}/{len(subset)}")
    print("Saved",out)
