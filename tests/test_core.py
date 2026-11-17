import cv2
import numpy as np
from photo_quality.core import (
    analyze_image, gamma_correct, adaptive_gamma, clahe_luminance,
    enhance_image, enhance_studio, rank_images, encode_jpeg, decode_image
)

def demo():
    x=np.tile(np.linspace(25,200,128,dtype=np.uint8),(128,1))
    return cv2.cvtColor(x,cv2.COLOR_GRAY2BGR)

def test_metrics_and_score():
    result=analyze_image(demo())
    assert 0<=result["quality_score"]<=100
    assert result["width"]==128

def test_brighten_gamma():
    img=demo()
    out=gamma_correct(img,0.6)
    assert out.mean()>img.mean()

def test_adaptive_gamma():
    img=(demo().astype(float)*0.3).astype(np.uint8)
    out,g=adaptive_gamma(img)
    assert 0<g<1 and out.mean()>img.mean()

def test_clahe_dimensions():
    assert clahe_luminance(demo()).shape==demo().shape

def test_rank_sorted():
    images={"dark":(demo()*0.25).astype(np.uint8),"normal":demo()}
    scores=[x["quality_score"] for x in rank_images(images)]
    assert scores==sorted(scores,reverse=True)

def test_auto_and_styles():
    img=demo()
    edited,meta=enhance_image(img)
    assert edited.shape==img.shape and "applied" in meta
    for style in ["Natural","Balanced","Vivid"]:
        out,data=enhance_studio(img,intensity=0.8,style=style)
        assert out.shape==img.shape and data["mean_absolute_change"]>=0

def test_zero_strength_and_codec():
    img=demo()
    out,_=enhance_studio(img,intensity=0)
    assert np.array_equal(out,img)
    assert decode_image(encode_jpeg(img)).shape==img.shape
