from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import pickle


def split_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_models(models, X_train, X_test, y_train):
    y_preds = []
    trained_models = []
    for model in models:
        trained_models.append(model.fit(X_train, y_train))
        y_preds.append(model.predict(X_test))
    return trained_models, y_preds

def evaluate_models(models_names, y_test, y_preds):
    model_results = []
    for model_name, y_pred in zip(models_names, y_preds):
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        precision_exited_1 = report['1']['precision']
        recall_exited_1 = report['1']['recall']
        f1_exited_1 = report['1']['f1-score']

        model_results.append([model_name, accuracy, precision_exited_1, recall_exited_1, f1_exited_1])
    results_df = pd.DataFrame(model_results, columns=['Model', 'Accuracy', 'Precision_Exited_1', 'Recall_Exited_1', 'F1_Exited_1'])
    results_df = results_df.sort_values(by='Recall_Exited_1', ascending=False).reset_index(drop=True)
    return results_df


def ptl_cm(models_names, y_test, y_preds):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    axes = axes.flatten() 

    for idx, (model_name, y_pred) in enumerate(zip(models_names, y_preds)):    
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            ax=axes[idx], 
            xticklabels=['Not Exited (0)', 'Exited (1)'], 
            yticklabels=['Not Exited (0)', 'Exited (1)']
        )
        
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_title(f'Confusion Matrix - {model_name}')
        
    plt.tight_layout()
    plt.show

    return plt

def save_model(trained_models, models_names, path):
    for trained_model, model_name in zip(trained_models, models_names):
        with open(f'{path}/{model_name}.pkl', 'wb') as file:  pickle.dump(trained_model, file)


