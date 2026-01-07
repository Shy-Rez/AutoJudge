import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR,"data")
MODELS_DIR=os.path.join(BASE_DIR,"models")

os.makedirs(MODELS_DIR,exist_ok=True)

df=pd.read_csv(os.path.join(DATA_DIR,"features_data.csv"))

numCol=["textLength","wordCount","key_graph","key_dp","key_greedy","key_bit","key_binary","key_tree","key_modulo","key_prime","key_lcm","key_probability","key_queue","key_stack","key_gcd","key_algorithm","key_prefix","key_suffix"]

X=df[["text"]+numCol]
y=df["problem_class"]

XTrain,XTest,yTrain,yTest=train_test_split(
    X,y,test_size=0.20,random_state=50,stratify=y
)

preprocessor=ColumnTransformer([
    ("text",TfidfVectorizer(
        max_features=12000, ngram_range=(1,2),
        min_df=5,max_df=0.85,sublinear_tf=True,
        stop_words="english"),"text"),
    ("num",StandardScaler(),numCol)])

def evaluate(model_name,yTest,yPred,class_labels):
    print(f"\n{model_name}")
    
    acc=accuracy_score(yTest,yPred)
    percent=acc*100
    print("Accuracy:",percent,"%")
    print()
    
    cm=confusion_matrix(yTest,yPred,labels=class_labels)
    cm_df=pd.DataFrame(cm,index=class_labels,columns=class_labels)
    
    print("Confusion Matrix:")
    print(cm_df)


svm_pipeline=Pipeline([
  ("preprocess",preprocessor),
  ("model",LinearSVC(
      C=0.15,max_iter=5000,
      class_weight={"easy":1.25,"medium":1.15,"hard":1.10}
    ))
])

svm_pipeline.fit(XTrain,yTrain)
yPred_svm=svm_pipeline.predict(XTest)

evaluate("Linear SVM",yTest,yPred_svm,svm_pipeline.classes_)

with open(os.path.join(MODELS_DIR,"clf_model.pkl"),"wb") as f:
  pickle.dump(svm_pipeline,f)

#Logistic Regression Model
# lr_pipeline=Pipeline([
#     ("preprocess",preprocessor),
#     ("model",LogisticRegression(
#       max_iter=1000,
#       class_weight={"easy":1.25,"medium":1.15,"hard":1.10},
#     ))
# ])

# lr_pipeline.fit(XTrain,yTrain)
# yPred_lr=lr_pipeline.predict(XTest)

# evaluate("Logistic Regression",yTest,yPred_lr,lr_pipeline.classes_)


#Random Forest Classifier Model
# rfc_pipeline=Pipeline([
#     ("preprocess",preprocessor),
#     ("model",RandomForestClassifier(
#         n_estimators=400,random_state=50,
#         class_weight={"easy":1.25,"medium":1.15,"hard":1.10},
#         n_jobs=-1
#     ))
# ])

# rfc_pipeline.fit(XTrain,yTrain)
# yPred_rf=rfc_pipeline.predict(XTest)

# evaluate("Random Forest",yTest,yPred_rf,rfc_pipeline.classes_)