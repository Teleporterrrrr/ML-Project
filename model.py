import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import itertools
import numpy as np
import pickle as pkl

class Model():
    def __init__(self, **kwargs):
        self.model = xgb.XGBClassifier(tree_method="hist", 
                                       objective="binary:logistic", 
                                       eval_metric="auc", 
                                       enable_categorical=True, 
                                       **kwargs)


    def fit(self, X, y):
        sampling_weight = (len(y)-sum(y["hospitaldischargestatus"]))/sum(y["hospitaldischargestatus"])
        print(sampling_weight)
        max_depth_list = list([6]) #6
        #sampling_weight_list = list(np.arange(0.5, sampling_weight+1, 1))
        sampling_weight_list = list([2.5]) #2.5
        subsample_list = list([0.8999]) #0.8999
        eta_list = list([0.2]) #0.2

        best_params = {}
        best_score = 0

        # Change this hyperparameter tuning to use K-Fold Cross Validation
        for params in list(itertools.product(max_depth_list, sampling_weight_list, subsample_list, eta_list)):
            max_depth, scale_pos_weight, subsample, eta = params
            model = xgb.XGBClassifier(tree_method="hist", 
                                       objective="binary:logistic", 
                                       eval_metric="auc", 
                                       enable_categorical=True, 
                                       max_depth=max_depth, 
                                       scale_pos_weight=scale_pos_weight, 
                                       subsample=subsample,
                                       eta=eta,
                                       seed=69420)  # you can add arguments as needed

            scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            
            score = sum(scores)/len(scores)

            if score > best_score:
                print("New best score: ", score)
                
                best_params = {'max_depth': max_depth, 'scale_pos_weight': scale_pos_weight, 'subsample':subsample, 'eta': eta}
                print("New best params: ", best_params)
                best_score = score


        # Once we have the optimal parameters, let's train on the full dataset instead of the train/val split
        print(best_params)
        print(best_score)
        self.model = xgb.XGBClassifier(tree_method="hist", 
                                       objective="binary:logistic", 
                                       eval_metric="auc", 
                                       enable_categorical=True, 
                                       max_depth = best_params["max_depth"],
                                       scale_pos_weight=best_params["scale_pos_weight"],
                                       subsample=best_params["subsample"],
                                       eta=best_params["eta"],
                                       seed=69420)
            
        self.model.fit(X, y)

        file_name = "xgb_model.pkl"

        # save
        pkl.dump(self.model, open(file_name, "wb"))

    def predict_proba(self, x):
        preds = self.model.predict_proba(x)[:, 1]

        return preds