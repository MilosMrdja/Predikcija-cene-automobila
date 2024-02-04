import json
from conf import *
from randomForest import *
from knn import *
from decisionTree import *

def get_the_best_random_forest(data:pd.DataFrame):
    f = open(FILE_JSON_FROM_SRC)

    dataj = json.load(f)
    model = dict()
    mera = 0
    acc_temp = 0
    for i in dataj["hiperp"]:
        m,acc = random_forest(data, n_tree=i["n_tree"])
        model[m] = acc
        if acc > acc_temp:
            acc_temp = acc
            mera = i["n_tree"]

    model = sorted(model.items(), key=lambda x:x[1], reverse=True)
    print(model)
    return model[0][0], model[0][1], mera
    
def get_the_best_knn(data):
    f = open(FILE_JSON_FROM_SRC)

    dataj = json.load(f)
    model = dict()
    mera = 0
    temp_acc = 0
    for i in dataj["hiperp"]:
        m,acc = knn(data, n=i["n_neighbors"])
        model[m] = acc
        if acc > temp_acc:
            temp_acc = acc
            mera = i["n_neighbors"]

    model = sorted(model.items(), key=lambda x:x[1], reverse=True)
    print(model)
    return model[0][0], model[0][1], mera

def get_the_best_decision_tree(data):
    f = open(FILE_JSON_FROM_SRC)

    dataj = json.load(f)
    model = dict()
    mera = 0
    temp_acc = 0
    for i in dataj["hiperp"]:
        m,acc = DecisionTree(data, md=i["max_depth_dt"])
        model[m] = acc
        if acc > temp_acc:
            temp_acc = acc
            mera = i["max_depth_dt"]

    model = sorted(model.items(), key=lambda x:x[1], reverse=True)
    print(model)
    return model[0][0], model[0][1], mera