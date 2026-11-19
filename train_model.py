"""Train four credit-default models and select a validation-cost policy."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
TARGET = 'default payment next month'
RAW = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE'] + [f'PAY_{i}' for i in (0,2,3,4,5,6)] + [f'BILL_AMT{i}' for i in range(1,7)] + [f'PAY_AMT{i}' for i in range(1,7)]
CATS = ['SEX','EDUCATION','MARRIAGE']
ENGINEERED = ['avg_bill_limit_ratio','max_bill_limit_ratio','late_payment_frequency','max_repayment_delay','avg_payment_amount','payment_variability','bill_change','aggregate_payment_bill_ratio']
NUMS = [c for c in RAW if c not in CATS] + ENGINEERED

class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Deterministic row features; fitted statistics live in later pipeline steps."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        d = X.copy()
        for c in RAW:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        d.loc[d.LIMIT_BAL <= 0, 'LIMIT_BAL'] = np.nan
        d.loc[~d.SEX.isin([1,2]), 'SEX'] = np.nan
        d['EDUCATION'] = d.EDUCATION.where(d.EDUCATION.isin([1,2,3]), 4)
        d['MARRIAGE'] = d.MARRIAGE.where(d.MARRIAGE.isin([1,2,3]), 4)
        d.loc[~d.AGE.between(18,100), 'AGE'] = np.nan
        delays = d[[f'PAY_{i}' for i in (0,2,3,4,5,6)]]
        for c in delays:
            d.loc[~d[c].between(-2,8), c] = np.nan
        bills = d[[f'BILL_AMT{i}' for i in range(1,7)]]
        pays = d[[f'PAY_AMT{i}' for i in range(1,7)]]
        for c in pays:
            d.loc[d[c] < 0, c] = np.nan
        limit = d.LIMIT_BAL
        ratios = bills.div(limit, axis=0)
        d['avg_bill_limit_ratio'] = ratios.mean(axis=1)
        d['max_bill_limit_ratio'] = ratios.max(axis=1)
        d['late_payment_frequency'] = delays.gt(0).sum(axis=1).div(delays.notna().sum(axis=1).replace(0,np.nan))
        d['max_repayment_delay'] = delays.max(axis=1)
        d['avg_payment_amount'] = pays.mean(axis=1)
        d['payment_variability'] = pays.std(axis=1).fillna(0)
        d['bill_change'] = d.BILL_AMT1 - d.BILL_AMT6
        d['aggregate_payment_bill_ratio'] = pays.sum(axis=1, min_count=1).div(bills.clip(lower=0).sum(axis=1, min_count=1).clip(lower=1))
        return d[NUMS+CATS].replace([np.inf,-np.inf],np.nan)

def load_data(path):
    d = pd.read_excel(path, header=1, engine='xlrd')
    if set(d.columns) != set(RAW+['ID',TARGET]) or len(d) == 0:
        raise ValueError('Unexpected UCI workbook schema; check header=1')
    if d.ID.isna().any() or d.ID.duplicated().any():
        raise ValueError('Missing or duplicate IDs')
    y = pd.to_numeric(d[TARGET], errors='raise')
    if y.isna().any() or not y.isin([0,1]).all():
        raise ValueError('Target must be binary and complete')
    return d[RAW], y.astype(int)

def make_pipeline(name):
    numeric = Pipeline([('impute',SimpleImputer(strategy='median', add_indicator=True)),('scale',StandardScaler())])
    categorical = Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))])
    prep = ColumnTransformer([('numeric',numeric,NUMS),('categorical',categorical,CATS)])
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1500, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, min_samples_leaf=100, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=160, min_samples_leaf=15, max_features='sqrt', n_jobs=-1, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=140, learning_rate=.06, max_depth=3, min_samples_leaf=40, random_state=42),
    }
    return Pipeline([('features',FeatureBuilder()),('preprocess',prep),('model',models[name])])

def evaluate(y, prob, threshold, fp_cost, fn_cost):
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'accuracy':float(accuracy_score(y,pred)), 'precision':float(precision_score(y,pred,zero_division=0)), 'recall':float(recall_score(y,pred,zero_division=0)), 'f1':float(f1_score(y,pred,zero_division=0)), 'roc_auc':float(roc_auc_score(y,prob)), 'pr_auc':float(average_precision_score(y,prob)), 'brier_score':float(brier_score_loss(y,prob)), 'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]], 'false_positives':int(fp), 'false_negatives':int(fn), 'business_cost':float(fp*fp_cost+fn*fn_cost)}

def optimize_threshold(y, prob, fp_cost, fn_cost):
    """Search every distinct validation score plus 0 and 1; ties favor higher cutoff."""
    y,prob = np.asarray(y),np.asarray(prob)
    thresholds = np.unique(np.r_[0., prob, 1.])
    order = np.argsort(prob)
    sorted_p, sorted_y = prob[order], y[order]
    positions = np.searchsorted(sorted_p, thresholds, side='left')
    positives_below = np.r_[0,np.cumsum(sorted_y)][positions]
    negatives_above = (len(y)-positions) - (sorted_y.sum()-positives_below)
    costs = negatives_above*fp_cost + positives_below*fn_cost
    best = np.flatnonzero(costs == costs.min())[-1]
    return float(thresholds[best])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=ROOT/'default of credit card clients.xls')
    p.add_argument('--output',type=Path,default=ROOT/'model.pkl')
    p.add_argument('--fp-cost',type=float,default=2000)
    p.add_argument('--fn-cost',type=float,default=20000)
    args=p.parse_args()
    if not np.isfinite([args.fp_cost,args.fn_cost]).all() or min(args.fp_cost,args.fn_cost)<=0: p.error('Costs must be finite and positive')
    X,y=load_data(args.data)
    X_tv,X_test,y_tv,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    X_train,X_val,y_train,y_val=train_test_split(X_tv,y_tv,test_size=.25,stratify=y_tv,random_state=42)
    print(f'Rows train={len(y_train)} validation={len(y_val)} test={len(y_test)}; default rate={y.mean():.4f}')
    print(f'Illustrative cost matrix [[TN,FP],[FN,TP]] = [[0,{args.fp_cost:g}],[{args.fn_cost:g},0]]')
    results=[]
    baseline=np.zeros(len(y_val))
    baseline_metrics=evaluate(y_val,baseline,.5,args.fp_cost,args.fn_cost)
    print('Majority baseline validation:',json.dumps(baseline_metrics))
    for name in ('Logistic Regression','Decision Tree','Random Forest','Gradient Boosting'):
        model=make_pipeline(name)
        cv=cross_val_score(model,X_train,y_train,cv=StratifiedKFold(3,shuffle=True,random_state=42),scoring='average_precision',n_jobs=1)
        model.fit(X_train,y_train)
        prob=model.predict_proba(X_val)[:,1]
        threshold=optimize_threshold(y_val,prob,args.fp_cost,args.fn_cost)
        m=evaluate(y_val,prob,threshold,args.fp_cost,args.fn_cost)
        results.append({'name':name,'model':model,'prob':prob,'threshold':threshold,'metrics':m,'cv_pr_auc_mean':float(cv.mean()),'at_050':evaluate(y_val,prob,.5,args.fp_cost,args.fn_cost)})
        print(f'{name:20} CV PR-AUC={cv.mean():.4f} val PR-AUC={m["pr_auc"]:.4f} ROC-AUC={m["roc_auc"]:.4f} accuracy@0.50={results[-1]["at_050"]["accuracy"]:.4f} accuracy@cost={m["accuracy"]:.4f} precision={m["precision"]:.4f} recall={m["recall"]:.4f} F1={m["f1"]:.4f} FP={m["false_positives"]} FN={m["false_negatives"]} cost={m["business_cost"]:,.0f} threshold={threshold:.5f}')
    selected=min(results,key=lambda r:(r['metrics']['business_cost'],-r['metrics']['pr_auc'],r['name']))
    test_prob=selected['model'].predict_proba(X_test)[:,1]
    test_metrics=evaluate(y_test,test_prob,selected['threshold'],args.fp_cost,args.fn_cost)
    test_at_050=evaluate(y_test,test_prob,.5,args.fp_cost,args.fn_cost)
    artifact={'pipeline':selected['model'],'model_name':selected['name'],'feature_names':RAW,'engineered_features':ENGINEERED,'threshold':selected['threshold'],'fp_cost':args.fp_cost,'fn_cost':args.fn_cost,'validation_labels':y_val.to_numpy(),'validation_probabilities':selected['prob'],'validation_metrics':selected['metrics'],'validation_at_050':selected['at_050'],'test_metrics':test_metrics,'test_at_050':test_at_050,'baseline_validation_metrics':baseline_metrics,'model_comparison':[{k:r[k] for k in ('name','threshold','metrics','cv_pr_auc_mean','at_050')} for r in results],'split_sizes':{'train':len(y_train),'validation':len(y_val),'test':len(y_test)},'random_state':42}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(artifact,args.output)
    print('Selected:',selected['name'],'validation threshold:',selected['threshold'])
    print('Validation at 0.50:',json.dumps(selected['at_050']))
    print('Validation optimized:',json.dumps(selected['metrics']))
    print('Untouched test at 0.50:',json.dumps(test_at_050))
    print('Untouched test at cost threshold:',json.dumps(test_metrics))
    print('Saved:',args.output)
if __name__=='__main__':
    import sys
    sys.modules['train_model'] = sys.modules['__main__']
    FeatureBuilder.__module__ = 'train_model'
    main()



