"""Experimental Richardson-Lucy deblurring benchmark.

Requires aligned controlled-blur input and reference images; blur kernels
must be KNOWN. This does not estimate real-world motion blur.

Input structure (not shipped in Git):
  01_coffee_before.jpg
  01_coffee_sharp_reference.jpg
  ...
See --help. The provided comparison images are separate artifacts.

Usage:
 python scripts/evaluate_deconvolution_iteration_07.py path/to/rating_images out.csv
"""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
from skimage.restoration import richardson_lucy

SAMPLES = [
    ("coffee","motion",19,0),
    ("chelsea","defocus",3.2,0),
    ("astronaut","motion",25,40),
    ("rocket","motion",23,-25),
    ("camera","defocus",4.5,0),
    ("grace_hopper","motion",19,20),
    ("grass","defocus",3.5,0),
    ("hubble_deep_field","motion",15,70),
]

def blur_kernel(kind, length, angle):
    if kind == "motion":
        n = int(length)
        kernel = np.zeros((n,n),np.float64)
        cv2.line(kernel,(0,n//2),(n-1,n//2),1,1)
        transform = cv2.getRotationMatrix2D((n//2,n//2),angle,1)
        kernel = cv2.warpAffine(kernel,transform,(n,n))
    else:
        n = int(round(length*6)) | 1
        axis = np.arange(n)-n//2
        profile = np.exp(-axis**2/(2*length**2))
        kernel = np.outer(profile,profile)
    return kernel/kernel.sum()

def deblur_known_psf(image, psf, iterations=8):
    """Experimental deconvolution with known PSF (not safe as an automatic filter)."""
    pad = max(psf.shape)//2
    src=cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_REFLECT_101)
    channels=[richardson_lucy(src[:,:,c].astype("float32")/255,
               psf,num_iter=iterations,clip=False) for c in range(3)]
    result=np.stack(channels,axis=2)
    return np.clip(result[pad:-pad,pad:-pad]*255,0,255).astype("uint8")

def errors(reference, output):
    diff=reference.astype("float64")-output.astype("float64")
    mse=float(np.mean(diff*diff))
    return round(float(np.mean(np.abs(diff))),3),round(
        float(10*np.log10(255**2/mse)) if mse else float("inf"),3)

def run(directory):
    rows=[]
    for idx,(name,kind,value,angle) in enumerate(SAMPLES,1):
        prefix=f"{idx:02d}_{name}_"
        inputs={}
        for method,label in (("before","Before"),("auto","Auto"),("balanced","Balanced")):
            inputs[label]=cv2.imread(str(directory/(prefix+method+".jpg")))
        ref=cv2.imread(str(directory/(prefix+"sharp_reference.jpg")))
        if ref is None or any(img is None for img in inputs.values()):
            raise FileNotFoundError("Missing input/reference for "+name)
        if ref.shape!=inputs["Before"].shape:
            ref=cv2.resize(ref,(inputs["Before"].shape[1],inputs["Before"].shape[0]))
        psf=blur_kernel(kind,value,angle)
        inputs["RL8"]=deblur_known_psf(inputs["Before"],psf,8)
        inputs["RL15"]=deblur_known_psf(inputs["Before"],psf,15)
        for method,image in inputs.items():
            mae,psnr=errors(ref,image)
            rows.append(dict(image=name,blur_type=kind,method=method,mae=mae,
                psnr=psnr,group="development" if idx<=4 else "secondary"))
    return rows

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("input_directory",type=Path)
    parser.add_argument("output_csv",type=Path)
    args=parser.parse_args()
    rows=run(args.input_directory)
    args.output_csv.parent.mkdir(parents=True,exist_ok=True)
    with args.output_csv.open("w",newline="") as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for group in ("development","secondary"):
        for method in ("Before","Auto","Balanced","RL8","RL15"):
            subset=[r for r in rows if r["group"]==group and r["method"]==method]
            print(group,method,"MAE:",round(np.mean([r["mae"] for r in subset]),2))
