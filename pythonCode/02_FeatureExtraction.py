import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR,"data")
os.makedirs(DATA_DIR,exist_ok=True)

df=pd.read_csv(os.path.join(DATA_DIR,"cleaned_data.csv"))

df["textLength"]=df["text"].str.len()
df["wordCount"]=df["text"].str.split().str.len()

keywords=["graph","dp","greedy","bit","lcm","modulo","prime",
          "probability","binary","tree","queue","stack",
          "gcd","algorithm","prefix","suffix"]

for key in keywords:
  colName=f"key_{key.replace(' ','_')}"
  df[colName]=df.apply(
    lambda row:(
      row["text"].lower().count(key)/row["wordCount"]
      if row["wordCount"]>0
      else 0
    ),
    axis=1
  )

numCol=[]
for col in df.columns:
  if col=="textLength":
    numCol.append(col)
  else:
    if col=="wordCount":
      numCol.append(col)
    else:
      if col.startswith("key_"):
        numCol.append(col)

df.to_csv(os.path.join(DATA_DIR,"features_data.csv"),index=False)