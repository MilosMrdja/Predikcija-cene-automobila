import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from conf import *
from util import pripremaPodataka


def random_forest(data:pd.DataFrame, n_tree):
    data = pd.get_dummies(data)
    y = np.array(data["price"])
    data = data.drop("price", axis=1)

    lista_kolona = list(data.columns)
    x = np.array(data)

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
    random_forest_model = RandomForestRegressor(n_estimators=n_tree, random_state=42)
    random_forest_model.fit(x_train, y_train)

    y_pred = random_forest_model.predict(x_test)
    '''
    errors = abs(y_pred - y_test)
    print('Mean Absolute Error:', round(np.mean(errors), 2),'degrees.')
    mape = 100 * (errors/y_test)
    acc = 100 - np.mean(mape)
    print("Accuracy: ",round(acc,2), '%')'''
    # vidimo da je preciznost mala, pa moramo da poboljsamo model
    rfmodel, acc, kolone = improve_rfmodel(random_forest_model, lista_kolona, x_train, x_test, y_train, y_test, n_tree)
    
    # crtanje grafika prediktovane i stvarne cene
    data_test = pd.read_csv(FILE_PATH_FROM_SRC_TEST)
    data_test = pripremaPodataka(data_test)
    data_test = pd.get_dummies(data_test,drop_first=True, dummy_na=False)
    y_test = np.array(data_test["price"])
    data_test = data_test.drop(columns=["price"], axis=1)
    data_test_encoded_aligned = data_test.reindex(columns=data.columns, fill_value=0)
    
    x_test = np.array(data_test_encoded_aligned)
    x_test = x_test[:, kolone]

    y_pred_test = rfmodel.predict(x_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
    plt.xlabel('Stvarne vrednosti')
    plt.ylabel('Predviđene vrednosti')
    plt.title('Stvarne vs. Predviđene vrednosti RandomForest modela')
    plt.show()


    return rfmodel, acc

def improve_rfmodel(rfmodel:RandomForestClassifier, kolone, x_train, x_test, y_train, y_test, n_tree):
    
    vaznosti = list(rfmodel.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(kolone,vaznosti)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    #[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances[:30]]

    rf_most_important = RandomForestRegressor(n_estimators= n_tree, random_state=42)
    important_indices = [kolone.index('model_year'), kolone.index('milage'),
                          kolone.index('Horsepower'),  kolone.index('Engine_power'),
                          kolone.index('transmission'),kolone.index('brand_BMW'),
                          kolone.index('brand_Porsche'),kolone.index('fuel_type_Diesel'),
                          kolone.index('fuel_type_Gasoline')]
    
    train_important = x_train[:, important_indices]
    test_important = x_test[:, important_indices]

    rf_most_important.fit(train_important, y_train)
    predictions = rf_most_important.predict(test_important) # predikcija
    errors = abs(predictions - y_test)
    #print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    mape = np.mean(100 * (errors / y_test))
    accuracy = 100 - np.mean(mape)
    #print('Accuracy:', round(accuracy, 2), '%.')



    # vaznosti promenljive grafik
    fig, ax = plt.subplots(figsize=(8, 4))

    # Crtanje grafika
    ax.bar(range(len(vaznosti[:15])), vaznosti[:15], orientation='vertical')

    # Podešavanje imena varijabli na x-osi
    imena = [v[0] for v in feature_importances[:15]]
    ax.set_xticks(range(len(vaznosti[:15])))
    ax.set_xticklabels(imena, rotation='vertical')

    # Oznake ose i naslov
    ax.set_ylabel('Vaznost')
    ax.set_xlabel('Promenljiva')
    ax.set_title('Grafik vaznosti promenljivih')

    # Prikazivanje grafa sa automatskim prilagođavanjem rasporeda
    plt.tight_layout()
    plt.show()



    return rf_most_important,accuracy, important_indices


    
