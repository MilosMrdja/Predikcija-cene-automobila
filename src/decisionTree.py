import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from conf import *
from util import pripremaPodataka

from sklearn.tree import DecisionTreeRegressor
def get_max_acc(data, md):
    x = data.drop(columns = ["price"])
    y = data["price"]  
    acc_best = 0
    best_model = None
    col_to_drop = []  
    columns = x.columns.tolist()

    for col in columns:
        new_data = data.drop(columns =[col])
        new_data = pd.get_dummies(new_data)
        x = new_data.drop(columns = ["price"])
        y = new_data["price"]

        
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
        dtree_regressor = DecisionTreeRegressor(max_depth=md)  # isprobano vise vrdnosti za k
        dtree_regressor.fit(x_train, y_train)
        acc_new = get_error_knn(dtree_regressor, x_val, y_val)
        if acc_new > acc_best:
            acc_best = acc_new
            best_model = dtree_regressor
            col_to_drop = col


    for i in range(len(columns)):
        for j in range(len(columns)):
            if i >= j:
                continue
            col1 = columns[i]
            col2 = columns[j]
            new_data = data.drop(columns =[col1, col2])
            new_data = pd.get_dummies(new_data)
            x = new_data.drop(columns = ["price"])
            y = new_data["price"]

            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
            dtree_regressor = DecisionTreeRegressor(max_depth=md)  # isprobano vise vrdnosti za k
            dtree_regressor.fit(x_train, y_train)

            acc_new = get_error_knn(dtree_regressor, x_val, y_val)
            #print(acc_new)
            if acc_new > acc_best:
                acc_best = acc_new
                best_model = dtree_regressor
                col_to_drop = [col1, col2]

    #print(acc_best, col_to_drop)
    return best_model,col_to_drop, acc_best


def DecisionTree(data,md):
    acc_best = 0
    model_best = None
    dropp_to = []
    for i in range(2):
        dtree_model, dropped_col, acc = get_max_acc(data,md)


        if acc > acc_best:
            if type(dropped_col)!=list:
                dropped_col = [dropped_col]
            data = data.drop(columns = dropped_col)
            acc_best = acc
            model_best = dtree_model
            for d in dropped_col:
                dropp_to.append(d)

    data = pd.get_dummies(data)

    data_test = pd.read_csv(FILE_PATH_FROM_SRC_TEST)
    data_test = pripremaPodataka(data_test)
    data_test = data_test.drop(columns=dropp_to)
    data_test = pd.get_dummies(data_test,drop_first=True, dummy_na=False)
    data_test_encoded_aligned = data_test.reindex(columns=data.columns, fill_value=0)

    x_test = data_test_encoded_aligned.drop(columns=["price"])
    y_test = data_test_encoded_aligned["price"]

    columns_test = x_test.columns.tolist()
    
    columns_test_filtered = [col for col in columns_test if not col in data.columns]

    y_pred_test = model_best.predict(x_test.drop(columns=columns_test_filtered))

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
    plt.xlabel('Stvarne vrednosti')
    plt.ylabel('Predviđene vrednosti')
    plt.title('Stvarne vs. Predviđene vrednosti DecisionTree modela')
    plt.show()

    return model_best, acc_best

def get_error_knn(model, x, y):
    y_pred = model.predict(x)

    errors = abs(y_pred - y)
    #print('Mean Absolute Error:', round(np.mean(errors), 2),'degrees.')
    mape = 100 * (errors/y)
    acc = 100 - np.mean(mape)
    #print("Accuracy: ",round(acc,2), '%') 
    return acc 