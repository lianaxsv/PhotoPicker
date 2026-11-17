"""PhotoPick: image quality gallery, ranking, and enhancement studio."""
from __future__ import annotations
import io
import zipfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from photo_quality.core import analyze_image, rank_images, decode_image, encode_jpeg, enhance_studio

st.set_page_config(page_title="PhotoPick · Photo Studio", page_icon="📷", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1350px;padding-top:2rem}
[data-testid="stMetric"] {background:#f5f6fa;border-radius:12px;padding:12px;border:1px solid #e5e7eb}
[data-testid="stImage"] img {border-radius:12px}
</style>""", unsafe_allow_html=True)

def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@st.cache_data
def sample_gallery():
    """Deterministic synthetic illustration of quality defects; not a real photo dataset."""
    h,w=540,800
    rng=np.random.default_rng(46)
    img=np.zeros((h,w,3),dtype=np.uint8)
    for y in range(h):
        img[y,:]=(int(210+25*y/h), int(170+45*y/h),int(75+55*y/h))
    cv2.circle(img,(620,120),65,(100,215,255),-1,cv2.LINE_AA)
    cv2.rectangle(img,(0,380),(w,h),(65,115,55),-1)
    for x in range(40,850,125):
        top=170+(x*7)%130
        cv2.rectangle(img,(x,top),(x+55,400),(95,135,125),-1)
        for yy in range(top+25,380,47):
            for xx in range(x+10,x+55,24):
                cv2.rectangle(img,(xx,yy),(xx+10,yy+22),(35,65,60),-1)
    cv2.ellipse(img,(220,390),(120,90),0,180,360,(30,105,20),-1)
    img=cv2.addWeighted(img,.97,rng.integers(0,255,size=img.shape,dtype=np.uint8),.03,0)
    return {
        "01 · Reference (synthetic)":img,
        "02 · Underexposed":np.clip(img.astype(np.float32)*.27,0,255).astype(np.uint8),
        "03 · Low contrast":np.clip(img.astype(np.float32)*.32+88,0,255).astype(np.uint8),
        "04 · Soft focus":cv2.GaussianBlur(img,(0,0),4),
        "05 · Overexposed":np.clip(img.astype(np.float32)*1.5+40,0,255).astype(np.uint8),
    }

st.title("📷 PhotoPick")
st.caption("Find your best photo. Understand quality issues. Compare corrections.")
with st.sidebar:
    st.header("Workspace")
    source=st.radio("Photos",["Sample gallery","Upload photos"])
    uploads=st.file_uploader("Upload images",type=["png","jpg","jpeg","webp"],accept_multiple_files=True) if source=="Upload photos" else []
    st.divider()
    st.header("Enhancement studio")
    preset=st.selectbox("Look",["Balanced","Natural","Vivid"])
    strength=st.slider("Correction strength",0,100,80,5)
    st.caption("Quality metrics are heuristics. Severe blur and clipped detail are not generally recoverable.")

if source=="Sample gallery":
    images=sample_gallery()
    st.info("This gallery uses **synthetic demonstration images**, not benchmark data. Upload your own photos for actual analysis.")
else:
    images={}
    for file in uploads or []:
        try:
            images[file.name]=decode_image(file.getvalue())
        except ValueError:
            st.warning(f"Could not decode {file.name}")
    if not images:
        st.info("Upload one or more photos in the sidebar.")
        st.stop()

records=rank_images(images)
names=list(images)
if st.session_state.get("photopick_selected") not in images:
    st.session_state.photopick_selected=records[0]["filename"]
selected=st.session_state.photopick_selected

a,b,c,d=st.columns(4)
a.metric("Photos",len(records))
b.metric("Best score",f"{records[0]['quality_score']:.0f}/100")
c.metric("Flagged",sum(bool(x["issues"]) for x in records))
d.metric("Selected rank",next(i for i,r in enumerate(records,1) if r["filename"]==selected))

st.subheader("Gallery")
st.caption("Photos ordered by heuristic quality score. Select any thumbnail to inspect it.")
for start in range(0,len(records),3):
    cols=st.columns(3)
    for col, record in zip(cols,records[start:start+3]):
        name=record["filename"]
        with col:
            st.image(rgb(images[name]),use_container_width=True)
            st.write(f"**{name}** · {record['quality_score']:.1f}/100")
            st.caption(", ".join(record["issues"]) or "No significant issues")
            if st.button("Selected ✓" if selected==name else "Open in studio",
                         disabled=selected==name,key="gallery_"+name,use_container_width=True):
                st.session_state.photopick_selected=name
                st.rerun()

st.divider()
st.subheader("Enhancement studio")
original=images[selected]
corrected,info=enhance_studio(original,intensity=strength/100,style=preset)
st.write(f"**{selected}** · {preset} · {strength}% strength")
comparison=st.radio("Compare",["Side by side","Reveal slider","Difference map"],horizontal=True)
if comparison=="Side by side":
    left,right=st.columns(2)
    left.image(rgb(original),caption="Before",use_container_width=True)
    right.image(rgb(corrected),caption="After",use_container_width=True)
elif comparison=="Reveal slider":
    reveal=st.slider("Enhanced portion (%)",0,100,50)
    split=int(original.shape[1]*reveal/100)
    combo=original.copy()
    combo[:,:split]=corrected[:,:split]
    cv2.line(combo,(split,0),(split,original.shape[0]-1),(255,255,255),max(2,original.shape[1]//200))
    st.image(rgb(combo),caption="Enhanced (left) / Original (right)",use_container_width=True)
else:
    diff=cv2.absdiff(original,corrected)
    boosted=np.clip(diff.astype(np.float32)*4,0,255).astype(np.uint8)
    left,right=st.columns(2)
    left.image(rgb(corrected),caption="Enhanced",use_container_width=True)
    right.image(rgb(boosted),caption="Pixel difference ×4",use_container_width=True)

m1,m2,m3=st.columns(3)
m1.metric("Mean pixel change",f"{info['mean_absolute_change']:.1f}/255")
m2.metric("Brightness",f"{info['after']['brightness']:.1f}",f"{info['after']['brightness']-info['before']['brightness']:+.1f}")
m3.metric("Heuristic score",f"{info['after']['quality_score']:.1f}",f"{info['after']['quality_score']-info['before']['quality_score']:+.1f}")
st.caption("A larger pixel change or heuristic score is not proof of better perceived image quality.")
st.write("**Issues:**",", ".join(info["before"]["issues"]) or "None")
st.write("**Applied:**"," → ".join(info["applied"]) or "None")
st.download_button("Download enhanced photo",encode_jpeg(corrected),file_name=Path(selected).stem+"_enhanced.jpg",mime="image/jpeg",type="primary")

st.divider()
st.subheader("Ranking details")
table=pd.DataFrame([{"Rank":i,"Photo":r["filename"],"Score":r["quality_score"],"Sharpness":r["sharpness"],
                     "Brightness":r["brightness"],"Contrast":r["contrast"],"Issues":", ".join(r["issues"]) or "None"}
                    for i,r in enumerate(records,1)])
st.dataframe(table,hide_index=True,use_container_width=True)
with st.expander("Download complete gallery"):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,img in images.items():
            edited,_=enhance_studio(img,intensity=strength/100,style=preset)
            archive.writestr(Path(name).stem+"_enhanced.jpg",encode_jpeg(edited))
        archive.writestr("rankings.csv",table.to_csv(index=False))
    st.download_button("Download ZIP with edited images and rankings",buf.getvalue(),
                       file_name="photopick_results.zip",mime="application/zip")
