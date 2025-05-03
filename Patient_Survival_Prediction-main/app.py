import gradio
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Response
from sklearn.metrics import accuracy_score, r2_score, precision_score, recall_score

save_file_name = "xgboost-model.pkl"
model = joblib.load(save_file_name)

# FastAPI object
app = FastAPI()

################################# Prometheus related code START ######################################################
import prometheus_client as prom

#acc_metric = prom.Gauge('titanic_accuracy_score', 'Accuracy score for few random 100 test samples')
#f1_metric = prom.Gauge('titanic_f1_score', 'F1 score for few random 100 test samples')
#precision_metric = prom.Gauge('titanic_precision_score', 'Precision score for few random 100 test samples')
#recall_metric = prom.Gauge('titanic_recall_score', 'Recall score for few random 100 test samples')

# LOAD TEST DATA
#pipeline_file_name = "xgboost-model.pkl"
#titanic_pipe= load_pipeline(file_name=pipeline_file_name)
#data = load_dataset(file_name=config.app_config.training_data_file)    # read complete data

test_data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
r2_metric = prom.Gauge('parient_survival_prediction_r2_score', 'R2 score for random 100 test samples') 

# X_train, X_test, y_train, y_test = train_test_split(                   # divide into train and test set
#     data[config.model_config.features],
#     data[config.model_config.target],
#     test_size=config.model_config.test_size,
#     random_state=config.model_config.random_state,
# )
# test_data = X_test.copy()
# test_data['target'] = y_test.values


# Function for updating metrics
def update_metrics(): 
    test = test_data.sample(100) 
    test_feat = test.drop('DEATH_EVENT', axis=1) 
    test_cnt = test['DEATH_EVENT'].values 
    #test_pred = make_prediction(input_data=test_feat)['predictions'] 
    test_pred = model.predict(test.iloc[:, :-1])
    r2 = r2_score(test_cnt, test_pred) 
    r2_metric.set(r2)

# def update_metrics():
#     global test_data
#     # Performance on test set
#     size = random.randint(100, 130)
#     test = test_data.sample(size, random_state = random.randint(0, 1e6))       # sample few 100 rows randomly
#     y_pred = model.predict(test.iloc[:, :-1])                           # prediction
#     acc = accuracy_score(test['target'], y_pred).round(3)                    # accuracy score
#     f1 = f1_score(test['target'], y_pred).round(3)                           # F1 score
#     precision = precision_score(test['target'], y_pred).round(3)             # Precision score
#     recall = recall_score(test['target'], y_pred).round(3)                   # Recall score
    
#     acc_metric.set(acc)
#     f1_metric.set(f1)
#     precision_metric.set(precision)
#     recall_metric.set(recall)

@app.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())

################################# Prometheus related code END ######################################################

# Function for prediction
def predict_death_event(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, 
                        high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame([{
        "age": age,
        "anaemia": anaemia,
        "creatinine_phosphokinase": creatinine_phosphokinase,
        "diabetes": diabetes,
        "ejection_fraction": ejection_fraction,
        "high_blood_pressure": high_blood_pressure,
        "platelets": platelets,
        "serum_creatinine": serum_creatinine,
        "serum_sodium": serum_sodium,
        "sex": sex,
        "smoking": smoking,
        "time": time
    }])
    # Predict using the trained model
    prediction = model.predict(input_data)
    
    # Optional: Return a human-readable result
    return "Death Event" if prediction[0] == 1 else "No Death Event"

# Gradio interface to generate UI link
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gradio.Interface(fn = predict_death_event,
                        inputs=[
                                    gradio.Number(label="Age"),
                                    gradio.Radio([0, 1], label="Anaemia (0=No, 1=Yes)"),
                                    gradio.Number(label="Creatinine Phosphokinase"),
                                    gradio.Radio([0, 1], label="Diabetes (0=No, 1=Yes)"),
                                    gradio.Number(label="Ejection Fraction"),
                                    gradio.Radio([0, 1], label="High Blood Pressure (0=No, 1=Yes)"),
                                    gradio.Number(label="Platelets"),
                                    gradio.Number(label="Serum Creatinine"),
                                    gradio.Number(label="Serum Sodium"),
                                    gradio.Radio([0, 1], label="Sex (0=Female, 1=Male)"),
                                    gradio.Radio([0, 1], label="Smoking (0=No, 1=Yes)"),
                                    gradio.Number(label="Follow-up Time")
                                ],
                        outputs = gradio.Textbox(type="text", label='Prediction', elem_id="out_textbox"),
                         title = title,
                         description = description,
                         allow_flagging='never'
                        )

# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gradio.mount_gradio_app(app, iface, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)