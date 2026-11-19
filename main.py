"""FastAPI inference from the locally trained model.pkl."""
from pathlib import Path
import math
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from train_model import RAW, optimize_threshold, evaluate

ARTIFACT = joblib.load(Path(__file__).resolve().parent/'model.pkl')
app = FastAPI(title='Credit Default Risk Prediction with Cost-Sensitive Decisions')

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    features: dict[str, float]

class CostRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    false_positive_cost: float = Field(default=2000, gt=0)
    false_negative_cost: float = Field(default=20000, gt=0)

def score(features):
    missing=set(RAW)-set(features)
    extra=set(features)-set(RAW)
    if missing or extra:
        raise HTTPException(422,detail={'missing':sorted(missing),'unexpected':sorted(extra)})
    if not all(math.isfinite(v) for v in features.values()):
        raise HTTPException(422,detail='All values must be finite')
    if features['LIMIT_BAL'] <= 0 or not 18 <= features['AGE'] <= 100 or features['SEX'] not in (1,2):
        raise HTTPException(422,detail='Invalid limit, age, or sex code')
    if any(features[f'PAY_AMT{i}'] < 0 for i in range(1,7)) or any(features[f'PAY_{i}'] not in range(-2,9) for i in (0,2,3,4,5,6)):
        raise HTTPException(422,detail='Invalid payment amount or repayment status')
    row=pd.DataFrame([{c:features[c] for c in RAW}])
    return float(ARTIFACT['pipeline'].predict_proba(row)[0,1])

@app.get('/health')
def health(): return {'status':'ok','model_loaded':True}

@app.get('/model-info')
def model_info():
    info = {k:ARTIFACT[k] for k in ('model_name','threshold','fp_cost','fn_cost','feature_names','engineered_features','validation_metrics','validation_at_050','test_metrics','test_at_050','baseline_validation_metrics','model_comparison','split_sizes')}
    info['test_accuracy_at_050'] = ARTIFACT['test_at_050']['accuracy']
    info['test_accuracy_at_cost_threshold'] = ARTIFACT['test_metrics']['accuracy']
    return info

@app.post('/predict')
def predict(request:PredictionRequest):
    probability=score(request.features)
    decision=probability>=ARTIFACT['threshold']
    return {'default_probability':probability,'predicted_class':int(decision),'risk_classification':'high' if decision else 'low','decision':'flag_for_review' if decision else 'no_flag','decision_threshold':ARTIFACT['threshold']}

@app.post('/optimize-threshold')
def optimize(request:CostRequest):
    fp,fn=request.false_positive_cost,request.false_negative_cost
    if not math.isfinite(fp) or not math.isfinite(fn): raise HTTPException(422,detail='Costs must be finite')
    y=ARTIFACT['validation_labels']; prob=ARTIFACT['validation_probabilities']
    threshold=optimize_threshold(y,prob,fp,fn)
    optimal=evaluate(y,prob,threshold,fp,fn)
    standard=evaluate(y,prob,.5,fp,fn)
    return {'optimal_threshold':threshold,'cost_matrix':[[0,fp],[fn,0]],'optimized_validation':optimal,'standard_050_validation':standard,'estimated_cost_reduction':standard['business_cost']-optimal['business_cost'],'note':'Costs and reductions are illustrative validation estimates. The saved /predict policy is unchanged.'}


