import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                             classification_report, precision_recall_curve, 
                             auc, confusion_matrix, f1_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



def resumetable(df):
    sh = df.shape
    print(f"Dataset Shape: {sh}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['% of Missing'] = summary['Missing'].apply(lambda x: f"{x / sh[0]:.2%}")
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    return summary

    
    return

def plot_dist(data, col):

    total = len(data)

    order = data[col].unique()

    plt.figure(figsize=(16,6))
    plt.suptitle(f"{col} Distribution", fontsize = 20)
    plt.subplot(1,2,1)
    p1 = sns.countplot(data= data, x = col, hue = col, order= order)
    p1.set_title(f"{col} Dist")
    p1.set_xlabel(f"{col} Categroy Name")
    for p in p1.patches:
        height = p.get_height()
        if height > 0:
            plt.text(p.get_x() + p.get_width()/2.,
                    height + 3,
                    f"{height/total*100:1.2f}%",
                    ha = "center", fontsize = 12)

    plt.subplot(1,2,2)
    p2 = sns.countplot(data= data, x = col, hue= "isFraud", order= order)
    p2.set_title(f"{col} Dist by Target")
    p2.set_xlabel(f"{col} Categroy Name")
    for p in p2.patches:
        height = p.get_height()
        if height > 0:
            plt.text(p.get_x() + p.get_width()/2.,
                    height + 3,
                    f"{height/total*100:1.2f}%",
                    ha = "center", fontsize = 12)
            
    tmp = pd.crosstab(data[col], data['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    pt = p2.twinx()
    pt = sns.pointplot(data = tmp, x = col, y = "Fraud", color= "black", alpha = 0.5, order=order)
    pt.set_ylabel("% of Fraud")


def evaluate_boosting_models(X_train, X_test, y_train, y_test, 
                             threshold=0.5,
                             xgb_params=None, lgb_params=None, cat_params=None,
                             models_to_run=["XGBoost", "LightGBM", "CatBoost"],
                             plot_pr_curve=True, 
                             train_output = False,
                             random_state=1234):
    """
    세 boosting 모델(XGBoost, LightGBM, CatBoost)을 학습시키고,  
    성능 평가표(accuracy, roc-auc, pr-AUC, precision/recall 테이블, confusion matrix, f1 score) 및 precision-recall curve를 출력하는 함수.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
        학습 데이터의 피처.
    X_test : array-like, shape (n_samples, n_features)
        테스트 데이터의 피처.
    y_train : array-like, shape (n_samples,)
        학습 데이터의 타깃.
    y_test : array-like, shape (n_samples,)
        테스트 데이터의 타깃.

    threshold : float, optional (default=0.5)
        확률 예측을 분류 레이블로 변환할 임계치.

    # 파라미터 전달은 딕셔너리 형태로

    xgb_params : dict, optional
        XGBoost classifier에 전달할 하이퍼파라미터 (default: 기본값).
    lgb_params : dict, optional
        LightGBM classifier에 전달할 하이퍼파라미터 (default: 기본값).
    cat_params : dict, optional
        CatBoost classifier에 전달할 하이퍼파라미터 (default: 기본값, verbose=0 설정 포함).


    plot_pr_curve : bool, optional (default=True)
        Precision-recall curve를 시각화할지 여부.
    random_state : int, optional (default=1234)
        랜덤시드.

    Returns
    -------
    results_df : pd.DataFrame
        각 모델의 평가 지표(accuracy, roc-auc, pr_auc, f1 score)를 정리한 DataFrame.
    porb_list : list
        pr-curve 그래프를 그리기 위한 예측확률 리스트
    """

    if xgb_params is None:
        xgb_params = {'eval_metric': 'logloss', 'random_state': random_state}
    if lgb_params is None:
        lgb_params = {'random_state': random_state}
    if cat_params is None:
        cat_params = {'verbose': 0, 'random_seed': random_state}

    all_models = {
        "XGBoost": XGBClassifier(**xgb_params),
        "LightGBM": LGBMClassifier(**lgb_params),
        "CatBoost": CatBoostClassifier(**cat_params)
    }

    results = []
    models = []
    if plot_pr_curve:
        plt.figure(figsize=(8, 6))

    for name in models_to_run:
        if name not in all_models:
            continue

        model = all_models[name]
        print(f"\n====== {name} ======")
        model.fit(X_train, y_train)

        if train_output:
            ## train data 평가지표
            y_prob_train = model.predict_proba(X_train)[:, 1]
            if name == "CatBoost":
                model.set_probability_threshold(threshold)
                y_pred_train = model.predict(X_train)
            else:
                y_pred_train = (y_prob_train >= threshold).astype(int)

            acc_train = accuracy_score(y_train, y_pred_train)
            roc_auc_train = roc_auc_score(y_train, y_prob_train)
            f1_train = f1_score(y_train, y_pred_train)
            precision_train, recall_train, _ = precision_recall_curve(y_train, y_prob_train)
            pr_auc_train = auc(recall_train, precision_train)

            print("\n=========Train data Information=========")
            print("Accuracy: {:.4f}".format(acc_train))
            print("ROC-AUC: {:.4f}".format(roc_auc_train))
            print("F1 Score: {:.4f}".format(f1_train))
            print("PR-AUC: {:.4f}".format(pr_auc_train))

            print("\nClassification Report on train data:\n", classification_report(y_train, y_pred_train))

            # Confusion Matrix 표 형태로 출력
            cm = confusion_matrix(y_train, y_pred_train)
            cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Pred 0', 'Pred 1'])
            print("\nConfusion Matrix:")
            print(cm_df)

        # test 데이터 평가지표
        y_prob = model.predict_proba(X_test)[:, 1]
        if name == "CatBoost":
                model.set_probability_threshold(threshold)
                y_pred = model.predict(X_test)
        else:
                y_pred = (y_prob >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)

        print("\n\n=========Test data Information=========")
        print("Accuracy: {:.4f}".format(acc))
        print("ROC-AUC: {:.4f}".format(roc_auc))
        print("F1 Score: {:.4f}".format(f1))
        print("PR-AUC: {:.4f}".format(pr_auc))

        print("\nClassification Report on test data:\n", classification_report(y_test, y_pred))

        # Confusion Matrix 표 형태로 출력
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Pred 0', 'Pred 1'])
        print("\nConfusion Matrix:")
        print(cm_df)

        if plot_pr_curve:
            plt.plot(recall, precision, lw=2, label=f'{name} (PR-AUC = {pr_auc:.4f})')

        results.append({
            'Model': name,
            'Accuracy': acc,
            'ROC-AUC': roc_auc,
            'F1 Score': f1,
            'PR-AUC': pr_auc
        })

        models.append(model)

    if plot_pr_curve:
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    results_df = pd.DataFrame(results).set_index('Model')
    return results_df, models


def pr_curve(X,y_true, models):

    for model in models:
        y_prob = model.predict_proba(X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(8,6))
        plt.plot(thresholds, precisions[:-1], label='Precision')
        plt.plot(thresholds, recalls[:-1], label='Recall')
        plt.xlabel("Threshold")
        plt.legend()
        plt.title(f'Precision-Recall vs Threshold on {model.__class__.__name__}')
        plt.grid(True)
        plt.show()


def plot_feature_importance(importance, features, top_n=20, figsize=(10, 8), title="Feature Importance"):
    # 데이터프레임으로 정리
    feat_imp_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })

    # 중요도 내림차순 정렬
    feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False).head(top_n)

    # 시각화
    plt.figure(figsize=figsize)
    plt.barh(feat_imp_df['Feature'][::-1], feat_imp_df['Importance'][::-1])  # 상위부터 아래로
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()