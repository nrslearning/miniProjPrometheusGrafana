# pull python base image
FROM python:3.10

# specify working directory
WORKDIR /Patient_Survival_Prediction

ADD requirements.txt .
ADD app.py .
ADD xgboost-model.pkl .
ADD heart_failure_clinical_records_dataset.csv .

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt



# # copy application files
# ADD /titanic_model_api/app/* ./app/

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app.py"]
