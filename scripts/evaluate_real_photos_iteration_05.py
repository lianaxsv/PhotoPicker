"""Re-run Iteration 5 on real sample photographs bundled with Python libraries.

These are genuine photos, not freshly web-scraped images. This experiment derives
synthetic degradations from 9 reference photographs. Reference-image fidelity is
not a human image-quality judgment. Requires matplotlib, scikit-image, pandas.

Run: python scripts/evaluate_real_photos_iteration_05.py
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage import data
from matplotlib import get_data_path
from photo_quality.core import analyze_image, enhance_image, enhance_studio

OUT=Path("results/iteration_05")
OUT.mkdir(parents=True,exist_ok=True)
SOURCES=("astronaut","coffee","chelsea","camera","rocket",
         "hubble_deep_field","retina","immunohistochemistry","grace_hopper")

def load(name):
    if name=="grace_hopper":
        image=cv2.imread(str(Path(get_data_path())/"sample_data"/"grace_hopper.jpg"))
    else:
        raw=getattr(data,name)()
        image=cv2.cvtColor(raw,cv2.COLOR_RGB2BGR if raw.ndim==3 else cv2.COLOR_GRAY2BGR)
    height,width=image.shape[:2]
    ratio=min(420/width,420/height)
    return cv2.resize(image,(round(width*ratio),round(height*ratio)),
                      interpolation=cv2.INTER_AREA)

def mae(a,b):
    return float(np.mean(np.abs(a.astype(np.float64)-b.astype(np.float64))))

def psnr(a,b):
    mse=float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
    return float("inf") if mse==0 else float(10*np.log10(255**2/mse))

def run():
    records=[]
    flags=[]
    for name in SOURCES:
        ref=load(name)
        rng=np.random.default_rng(5810+sum(map(ord,name)))
        v={
          "unchanged":ref,
          "dark_x0.45":np.uint8(np.clip(ref.astype("float32")*.45,0,255)),
          "dark_x0.7":np.uint8(np.clip(ref.astype("float32")*.7,0,255)),
          "bright_x1.5_plus25":np.uint8(np.clip(ref.astype("float32")*1.5+25,0,255)),
          "blur_sigma2.5":cv2.GaussianBlur(ref,(0,0),2.5),
          "low_contrast":np.uint8(np.clip((ref.astype("float32")-127)*.3+127,0,255)),
          "noise_sigma18":np.uint8(np.clip(ref.astype("float32")
              +rng.normal(0,18,ref.shape),0,255)),
        }
        for condition,image in v.items():
            m=analyze_image(image)
            flags.append(dict(image=name,condition=condition,issues="|".join(m["issues"]),
                underexposed=int("underexposed" in m["issues"]),
                overexposed=int("overexposed" in m["issues"]),
                possible_blur=int("possibly_blurry" in m["issues"]),
                low_contrast=int("low_contrast" in m["issues"])))
            outputs={"input":image,"auto":enhance_image(image)[0],
                     "balanced":enhance_studio(image,intensity=.8,style="Balanced")[0]}
            for method,output in outputs.items():
                records.append(dict(image=name,condition=condition,method=method,
                     mae_to_reference=round(mae(ref,output),4),
                     psnr_db=round(psnr(ref,output),3) if np.isfinite(psnr(ref,output)) else None,
                     pixel_change=round(mae(image,output),3)))
    frame=pd.DataFrame(records)
    frame.to_csv(OUT/"metrics.csv",index=False)
    pd.DataFrame(flags).to_csv(OUT/"flags.csv",index=False)
    print(frame.groupby(["condition","method"]).mae_to_reference.mean().round(2))
    print("Wrote",len(frame),"method measurements from",len(SOURCES),"sources")
if __name__=="__main__":
    run()
