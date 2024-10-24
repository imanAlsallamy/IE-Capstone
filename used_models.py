from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

models_names = [    
    'Logistic_Regression',
    'Random_Forest',
    'Decision_Tree',
    'XGB_Classifier',
    'SVC'
]

models = [
    LogisticRegression(), 
    RandomForestClassifier(random_state=42),
    DecisionTreeClassifier(),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    SVC()
]