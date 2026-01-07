import pandas as pd
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR,"data")
os.makedirs(DATA_DIR,exist_ok=True)

with open(os.path.join(DATA_DIR,"problems_data.jsonl"),"r",encoding="utf-8") as f:
    df=pd.read_json(f,lines=True)

textColumns=["title","description","input_description","output_description","sample_io"]

for col in textColumns:
    df[col]=df[col].fillna("")
    df[col]=df[col].apply(lambda x:str(x))

df["text"]=(df["title"]+" "+df["description"]+" "+df["input_description"]+" "+df["output_description"]+" "+df["sample_io"])

df["text"]=df["text"].str.lower()
df["text"]=df["text"].str.replace("\n"," ")
df["text"]=df["text"].str.replace("\t"," ")
df["text"]=df["text"].apply(lambda x:" ".join(x.split()))

print(df["problem_class"].value_counts())
print()
print(df["problem_score"].describe())
print()

df.to_csv(os.path.join(DATA_DIR,"cleaned_data.csv"),index=False)

# To plot Class and Score distribution graphs
# plt.figure()
# df["problem_class"].value_counts().plot(kind="bar")
# plt.title("Class Distribution")
# plt.xlabel("Problem Class")
# plt.ylabel("Number of problems")
# plt.show()

# plt.figure()
# plt.hist(df["problem_score"],bins=50)
# plt.title("Score Distribution")
# plt.xlabel("Problem Score")
# plt.ylabel("Number of problems")
# plt.show()