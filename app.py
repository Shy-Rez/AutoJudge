import pickle
import pandas as pd
import streamlit as st
import os

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR,"models")
with open(os.path.join(MODELS_DIR,"clf_model.pkl"),"rb") as f:
    clf_model=pickle.load(f)

with open(os.path.join(MODELS_DIR,"reg_model.pkl"),"rb") as f:
    reg_model=pickle.load(f)

print("BASE_DIR:", BASE_DIR)
print("MODELS_DIR:", MODELS_DIR)

keywords=["graph","dp","greedy","bit","lcm","modulo","prime","probability","binary","tree","queue","stack",
"gcd","algorithm","prefix","suffix"]

numCol=["textLength","wordCount","key_graph","key_dp","key_greedy","key_bit","key_binary","key_tree","key_modulo","key_prime","key_lcm","key_probability","key_queue","key_stack","key_gcd","key_algorithm","key_prefix","key_suffix"]

def extract(text):
    data={}
    data["text"]=text
    data["textLength"]=len(text)
    data["wordCount"]=len(text.split())
    
    lower=text.lower()
    if data["wordCount"]==0:
        wc=1
    else:
        wc=data["wordCount"]
    for k in keywords:
        col="key_"+k.replace(" ","_")
        data[col]=lower.count(k)/wc

    return pd.DataFrame([data])

st.set_page_config(layout="wide")
st.title("AutoJudge")
st.write("Programming Problem Difficulty Predictor")

title=st.text_input(
    label="Problem Title:",
    placeholder="Enter the problem title...")
desc=st.text_area(
    label="Problem Description:",
    placeholder="Enter the problem statement...",
    height=120)
input=st.text_area(
    label="Input Description:",
    placeholder="Describe the input format...",
    height=120)
output=st.text_area(
    label="Output Description:",
    placeholder="Describe the output format...",
    height=120)
samplein=st.text_area(
    label="Sample Input:",
    placeholder="Provide the sample input...",
    height=100)
sampleout=st.text_area(
    label="Sample Output:",
    placeholder="Provide the sample output...",
    height=100)

if st.button("Predict Difficulty"):
    combinedText=title+" "+desc+" "+input+" "+output+" "+samplein+" "+sampleout

    if len(combinedText.replace(" ",""))==0:
        st.warning("Please enter a problem description")
    else:
        X=extract(combinedText)
        X=X[["text"]+numCol]

    pclass=clf_model.predict(X)[0]
    pscore=reg_model.predict(X)[0]

    scale={"easy":0.0,"medium":0.8,"hard":1.6}
    pscore=pscore+scale[pclass]
    pscore=max(1.0,min(10.0,pscore))

    st.success(f"Difficulty Class:{pclass}")
    st.info(f"Difficulty Score:{pscore:.2f}")
