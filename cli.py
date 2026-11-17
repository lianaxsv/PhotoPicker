"""Batch image enhancement: python cli.py input_folder output_folder --mode auto."""
import argparse
import csv
from pathlib import Path
import cv2
from photo_quality.core import analyze_image, enhance_image

def main():
    parser=argparse.ArgumentParser(description="Batch analyze and enhance photographs")
    parser.add_argument("input",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--mode",choices=["auto","gamma","clahe","sharpen","none"],default="auto")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in sorted(args.input.iterdir()):
        if path.suffix.lower() not in {".png",".jpg",".jpeg",".webp"}:
            continue
        image=cv2.imread(str(path))
        if image is None:
            print("Skipping unreadable file:",path)
            continue
        enhanced,info=enhance_image(image,args.mode)
        output=args.output/(path.stem+"_enhanced.jpg")
        cv2.imwrite(str(output),enhanced)
        rows.append({"filename":path.name,**info["before"],"output":str(output),"applied":"; ".join(info["applied"])})
    if rows:
        with (args.output/"results.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Processed {len(rows)} images")

if __name__=="__main__":
    main()
