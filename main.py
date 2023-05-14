# from project_utils import create_data_for_project

# data = create_data_for_project(".")

import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np

from data import load_data, preprocess_x, split_data
from parser import parse
from model import Model
import pickle as pkl


def main():
    args = parse()

    x = load_data("train_x.csv")
    y = load_data("train_y.csv")
  
    # Add rate of change variables to dataframe
    processed_x = preprocess_x(x)

    y = load_data("train_y.csv")
    processed_x['ethnicity'] = processed_x['ethnicity'].astype("category")
    processed_x['gender'] = processed_x['gender'].astype("category")

    processed_x.to_csv("train_data.csv")


    processed_x = pd.read_csv("train_data.csv")
    processed_x['ethnicity'] = processed_x['ethnicity'].astype("category")
    processed_x['gender'] = processed_x['gender'].astype("category")
    processed_x['patientunitstayid'] = processed_x['patientunitstayid'].astype(np.int64)
    y = load_data("train_y.csv")
    
    y = y.set_index('patientunitstayid')
    y = y.reindex(index=processed_x['patientunitstayid'])
    y = y.reset_index(drop=True)
    processed_x.drop("patientunitstayid", axis=1, inplace=True)
    
    model = Model()
    print(y)
    model.fit(processed_x, y)

    x_test = load_data("test_x.csv")

    """"
    processed_x_test = preprocess_x(x_test)
    processed_x_test['ethnicity'] = processed_x_test['ethnicity'].astype("category")
    processed_x_test['gender'] = processed_x_test['gender'].astype("category")
    processed_x_test.to_csv("test_data.csv")
    """

    processed_x_test = pd.read_csv("test_data.csv")
    processed_x_test['ethnicity'] = processed_x_test['ethnicity'].astype("category")
    processed_x_test['gender'] = processed_x_test['gender'].astype("category")
    prediction_probs = model.predict_proba(processed_x_test.drop("patientunitstayid", axis=1))

    processed_x_test["patientunitstayid"] = processed_x_test["patientunitstayid"].astype("int32")
    output = pd.DataFrame({"patientunitstayid": processed_x_test["patientunitstayid"].values.tolist(), "hospitaldischargestatus": prediction_probs})

    output.to_csv("submission.csv", index=False, header=True)

if __name__ == "__main__":
    main()
