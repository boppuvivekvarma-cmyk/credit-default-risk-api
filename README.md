# Credit Default Risk Prediction with Cost-Sensitive Decisions

Educational model using the UCI Default of Credit Card Clients workbook. The three main files are `train_model.py`, generated `model.pkl`, and `main.py`.

## Run in the activated VS Code PowerShell terminal

```powershell
python -m pip install -r requirements.txt
python train_model.py
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Training generates `model.pkl` and prints the measured model comparison. Open http://127.0.0.1:8000/docs for Swagger. In another terminal, test:

```powershell
$features = @{ LIMIT_BAL=20000; SEX=2; EDUCATION=2; MARRIAGE=1; AGE=24; PAY_0=2; PAY_2=2; PAY_3=-1; PAY_4=-1; PAY_5=-2; PAY_6=-2; BILL_AMT1=3913; BILL_AMT2=3102; BILL_AMT3=689; BILL_AMT4=0; BILL_AMT5=0; BILL_AMT6=0; PAY_AMT1=0; PAY_AMT2=689; PAY_AMT3=0; PAY_AMT4=0; PAY_AMT5=0; PAY_AMT6=0 }
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body (@{features=$features} | ConvertTo-Json)
Invoke-RestMethod http://127.0.0.1:8000/optimize-threshold -Method Post -ContentType 'application/json' -Body (@{false_positive_cost=2000;false_negative_cost=20000} | ConvertTo-Json)
```

The workbook is at the project root. Override with `python train_model.py --data 'path\to\file.xls'`. Header row 2 is read with `header=1`. ID is excluded. The script checks the exact source columns, duplicate IDs, and binary target. Unknown education and marriage codes are grouped as other. Impossible limits, ages, delay codes, and negative payment amounts become missing and are imputed from training data. Negative bills are retained because a credit balance is possible. Behavioural features are built from the six historical months. Median imputation, scaling, and category encoding are fitted in the model pipeline only on training folds.

A fixed stratified 60/20/20 train/validation/test split uses seed 42. Three-fold stratified training cross-validation reports mean PR-AUC. Validation cost selects the model and threshold; the test set is evaluated once. The majority-class baseline always predicts no default. The illustrative matrix is `[[TN=0, FP=2000], [FN=20000, TP=0]]`; business cost is `2000*FP + 20000*FN`. The threshold search checks every distinct validation probability and endpoints 0 and 1. `/optimize-threshold` repeats this using stored validation labels and probabilities for supplied costs. It reports an estimated cost reduction from the 0.50 policy. `/predict` continues using the saved default policy until retraining.

`GET /health` checks startup, `GET /model-info` returns measured scores and input schema, `POST /predict` scores one account, and `POST /optimize-threshold` analyzes a new cost assumption. Inputs require all 23 source predictors with no extra fields. Risk classification means whether the selected policy flags the account for review; it is not a lending decision.

Accuracy is the share of all correct decisions. Default recall is the share of actual defaults caught. ROC-AUC measures ranking across false-positive rates; PR-AUC emphasizes ranking of defaults under imbalance. Business cost applies the stated error costs at a specific threshold. Brier score measures probability error; lower is better. The data is historical and may not represent current lending or another region. Cost units are hypothetical, and validation cost gains may differ on new customers. Do not deploy for lending without contemporary validation, calibration review, fairness review, and human oversight. Load only a trusted locally generated pickle artifact.

## Mentor deployment demonstration

The model is already trained: `model.pkl` contains the fitted Gradient Boosting pipeline, validation decisions, and measured test scores. Start FastAPI in one PowerShell terminal:

```powershell
cd C:\Users\vivekvarma\Documents\SDC
.\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

For a temporary public URL, start a second terminal:

```powershell
cd C:\Users\vivekvarma\Documents\SDC
.\tools\cloudflared.exe tunnel --url http://127.0.0.1:8000
```

Copy the printed `https://...trycloudflare.com` URL. The URL is public, changes whenever the tunnel restarts, and works only while both terminals remain open. The quick tunnel is for a class demonstration, not permanent hosting. To test in Postman, import `Credit_Default_Risk.postman_collection.json`, update its `baseUrl` collection variable to the latest tunnel URL, and send each of the four saved requests. In the Postman **Body** tab, POST requests use **raw → JSON**. Save screenshots of the 200 responses and the Postman test results for the mentor.

## Public deployment

Public API: https://credit-default-risk-api-vivek.onrender.com/docs
Public source repository: https://github.com/boppuvivekvarma-cmyk/credit-default-risk-api

The API is deployed as a Docker web service on Render's Free plan. The Docker image includes `main.py`, `train_model.py`, and the saved `model.pkl`, and pins the Python packages used to load the model. Render starts Uvicorn with its assigned port and checks `/health`. The service URL is stable, but Free instances can sleep after inactivity; the first request may take longer while it wakes. The raw `.xls` dataset and local virtual environments are not in the public repository. To retrain elsewhere, obtain the UCI Default of Credit Card Clients workbook and place it at the project root, then run `python train_model.py` after installing `requirements.txt`.

For Postman, import `Credit_Default_Risk.postman_collection.json`; its `baseUrl` points to the Render service. Send the health, model-info, predict, and optimize-threshold requests. The public endpoint is an educational demonstration, not a production lending service.
