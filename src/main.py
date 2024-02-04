import pandas as pd
from conf import *
from util import *
import seaborn as sb
from linearRegresion import linear_regresion_model
from decisionTree import DecisionTree
from hiperparametri import *

if __name__ == "__main__":
    res = {}
    data = pd.read_csv(FILE_PATH_FROM_SRC_TRAIN)
    data = pripremaPodataka(data=data)
    print("*************************************")

    decision_tree_model, acc_dt, md_dtree = get_the_best_decision_tree(data)
    random_forest_model, acc_rf, n_tree = get_the_best_random_forest(data)
    lin_reg_model, acc_lr = linear_regresion_model(data)
    knn_model,acc_knn, n_ne = get_the_best_knn(data)
    
    res[decision_tree_model] = acc_dt
    res[random_forest_model] = acc_rf
    res[lin_reg_model] = acc_lr
    res[knn_model] = acc_knn
    print("******************************************************************")
    for k,v in res.items():
        print(f"Algoritam: {k}, ima preciznost od: {v} %.\n")
    print("******************************************************************")
    sorted_dict = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
    print(f"Zakljucak  >>\tNajbolji model koji koristimo za ovaj set podataka za predikciju cene automobila jeste: {next(iter(sorted_dict))}\n")

    print(list(res.keys()))
    keys = ["DecisionTree, max_depth:" + str(md_dtree),"RandomForest, N_tree:" + str(n_tree), "LinearRegression", "KNeighbors, K:"+str(n_ne)]
    values = list(res.values())

    # Crtanje grafika
    plt.figure(figsize=(10, 6))
    sb.barplot(x=keys, y=values, color='blue')
    plt.xlabel('Model')
    plt.ylabel('Vrednost %')
    plt.title('Grafik modela i procenat tacnosti')
    plt.show()

 