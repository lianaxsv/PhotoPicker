"""Generate PhotoPick blur-rating comparisons from the CC0 Kwentar Blur Dataset.

Download the dataset yourself from:
https://www.kaggle.com/datasets/kwentar/blur-dataset
and unzip to a local folder with 'motion_blurred', 'defocused_blurred', 'sharp'.

Usage: python scripts/make_kwentar_blur_review.py /path/to/data /path/to/output --max-per-class 8

Output: numbered side-by-side JPEGs and rating_manifest.csv. The sharp image is a
different camera capture of the same scene, not an aligned pixel ground truth:
do not calculate image-pair PSNR/MAE from it.

Requires: pip install -r requirements.txt
"""
from pathlib import Path
import argparse,csv
import cv2
import numpy as np
from photo_quality.core import enhance_image, enhance_studio, analyze_image

def parse_key(p):
    # Stem example 106_NIKON-D3400-35MM_M
    parts=p.stem.rsplit("_",1)
    return parts[0] if len(parts)==2 else p.stem

def image_tile(bgr, height=340, width=400):
    h,w=bgr.shape[:2]
    scale=min(width/w,height/h)
    resized=cv2.resize(bgr,(max(1,int(w*scale)),max(1,int(h*scale))),
                       interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
    result=np.full((height,width,3),245,dtype=np.uint8)
    rh,rw=resized.shape[:2]
    result[(height-rh)//2:(height-rh)//2+rh,(width-rw)//2:(width-rw)//2+rw]=resized
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("dataset",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--max-per-class",type=int,default=8)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for folder in ("motion_blurred","defocused_blurred"):
        root=args.dataset/folder
        if not root.is_dir():
            raise SystemExit(f"Missing folder: {root}")
        photos=sorted(p for p in root.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"})
        for path in photos[:max(0,args.max_per_class)]:
            original=cv2.imread(str(path))
            if original is None:continue
            auto,meta=enhance_image(original)
            balanced,other=enhance_studio(original,intensity=.8,style="Balanced")
            blocks=[image_tile(x) for x in [original,auto,balanced]]
            sheet=np.hstack(blocks)
            cv2.putText(sheet,"BEFORE",(12,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(10,10,10),2)
            cv2.putText(sheet,"AUTO",(412,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(10,10,10),2)
            cv2.putText(sheet,"BALANCED",(812,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(10,10,10),2)
            name=f"{len(rows)+1:03d}_{folder}_{path.stem}.jpg"
            cv2.imwrite(str(args.output/name),sheet)
            rows.append(dict(id=len(rows)+1,source=path.name,blur_type=folder,
                             output=name,detected_issues="|".join(meta["before"]["issues"]),
                             auto_applied="|".join(meta["applied"]),
                             before_score=meta["before"]["quality_score"],
                             auto_score=meta["after"]["quality_score"],
                             balanced_score=other["after"]["quality_score"],
                             human_preference="",human_notes=""))
    with (args.output/"rating_manifest.csv").open("w",newline="") as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys())
            writer.writeheader();writer.writerows(rows)
    print(f"Wrote {len(rows)} comparisons to {args.output}. No photo files are uploaded.")

if __name__=="__main__":main()
