import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import re

from sklearn.model_selection import train_test_split # potrebni regex za izdvanje podataka

def brandKolona(data):
    print("\n***********************************\n")
    print(data.brand.value_counts()) # ispisali smo da vidimo vrednosti, sve vrednosti su validne
    
def modelKolona(data):
    print("\n***********************************\n")
    print(data.model.value_counts()) # posto ih ima previse ispitacemo samo za znak "-"
    if "–" in data["model"].values or data["model"].isnull().sum()>0:
        print("\nPostoje nevalidne vrednosti")
    else:
        print("\nSve vrednosti su validne")

def modelYearKolona(data):
    print("\n***********************************\n")
    print(data.model_year.value_counts()) # za model_year ima malo vrednosti, i mozemo rucno videti da je sve validno

def milageKolona(data):
    print("\n***********************************\n")
    print(data.milage.value_counts()) # ima puno vrednosti pa cemo samo proveriti kao i za model
    if "–" in data["milage"].values or data["milage"].isnull().sum()>0:
        print("\nPostoje nevalidne vrednosti")
    else:
        print("\nSve vrednosti su validne")

def fuelTypeKolona(data):
    print("\n***********************************\n")
    print(data.fuel_type.value_counts())
    sb.countplot(data=data, x="fuel_type")
    plt.show()

# funkcija koja prima tekst iz kolone engine i vraca dve vrednosti koje treba da stoje u koloni HorsePorwer i Engine
# ako se u koloni engine ne nalaze potrebni podaci, onda postavljamo na null i kasnije rukujemo njima
def izdvojHPiE(tekst):
    horsepower = re.search(r'(\d+\.\d+)HP|\d+\.\d+', tekst)
    engine_power = re.search(r'(\d+\.\d+L|\d+\.\d+ Liter)', tekst)
    return horsepower.group(1) if horsepower else pd.NA, engine_power.group(1) if engine_power else pd.NA

def enginePowerKolona(data):
    print("\n***********************************\n")
    print(data.Engine_power.value_counts()) # vidimo da su sve vrednosti validne

def horsePowerKolona(data):
    print("\n***********************************\n")
    print(data.Horsepower.value_counts()) #posto ima puno vrednosti moracemo proveriti
    check = data["Horsepower"].isnull().sum() 
    if check > 0:
        print("\nPostoje nevalidne vrednosti")
    else:
        print("\nSve vrednosti su validne")

def transmissionKolona(data):
    print("\n***********************************\n")
    print(data.transmission.value_counts()) #mozemo rucno videti da su sve validne

def accidentKolona(data):
    print("\n***********************************\n")
    print(data.accident.value_counts()) #mozemo rucno videti da su sve validne

def intColorKolona(data):
    print("\n***********************************\n")
    print(data.int_col.value_counts())

def extColorKolona(data):
    print("\n***********************************\n")
    print(data.ext_col.value_counts()) # ima ih previse da vidimo rucno
    if "–" in data["ext_col"].values or data["ext_col"].isnull().sum()>0:
        print("\nPostoje nevalidne vrednosti")
    else:
        print("\nSve vrednosti su validne")

def priceKolona(data):
    print("\n***********************************\n")
    print(data.price.value_counts()) # ima ih previse da vidimo rucno
    if "–" in data["price"].values or data["price"].isnull().sum()>0:
        print("\nPostoje nevalidne vrednosti")
    else:
        print("\nSve vrednosti su validne")

# ako je sve ispravno odnosno nema None vrednosti vraca true, u suprotnom false
def proveriNoneVrednosti(data):
    for ime_kolone in data.columns:
        if data[ime_kolone].isnull().sum()>0:
            return False
    return True


    

def pripremaPodataka(data,ispisVrednosti=False):
    print(data.info())
    # ispisom ovih info vidimo koliko je vrednosti null, koje kolone sadrzi nas fajl i tipove vrednosti
    # Podaci su preuzeti sa interneta stoga cemo raditi sa svim kolona sem kolone "clean_title", koje cemo samo izbaciti, "dropovati"
    data = data.drop(columns = ["clean_title"])
    print(data.columns)

    # vrsimo provere nad null vrednostima i onima koji imaju vrednost "-", takav je fajl koji smo dobili
    # vrednosti "-" treba prebaciti u null
    if ispisVrednosti:
        brandKolona(data)
        modelKolona(data)
        modelYearKolona(data)
        milageKolona(data)

    # milage - uklanjenje sve osim broja
    data["milage"] = data["milage"].apply(lambda x: x.split(' ')[0])
    data["milage"] = data["milage"].apply(lambda x: x.replace(',', ''))
    data["milage"] = data["milage"].astype('float')
    

    # fuel_type - popunjavanje srednom vrednoscu
    data["fuel_type"].replace("–",pd.NA,inplace=True)
    if ispisVrednosti:
        fuelTypeKolona(data)
    data.fuel_type.fillna("Gasoline", inplace = True)


    # engine - grupisanje u dve kolone, konjska snaga i snaga motora, brisanje kolone engine
    # podaci su grupisani pre "HP" i "L", tako da cemo ih izdvojiti i splitovati u dve kolone
    data[['Horsepower', 'Engine_power']] = data['engine'].apply(izdvojHPiE).apply(pd.Series)
    
    # HP - popunjavanje srednom vrednoscu
    if data["Horsepower"].isnull().sum() > 0:
        data.Horsepower.fillna(0, inplace = True)
    data["Horsepower"] = data["Horsepower"].astype('float')
    suma =data["Horsepower"].sum()
    brojPodatak = data["Horsepower"].count()
    data.Horsepower.replace(0,suma/brojPodatak, inplace = True)

    # engine_power - popunjavanje srednom vrednoscu
    if data["Engine_power"].isnull().sum() > 0:
        data.Engine_power.fillna(0, inplace = True)
    data['Engine_power'] = data['Engine_power'].replace({r'[^\d.]': ''}, regex=True).astype(float)
    suma = data["Engine_power"].sum()
    brojPodatak = data["Engine_power"].count()
    data.Engine_power.replace(0,suma/brojPodatak, inplace = True)

    data = data.drop(columns=["engine"])


    # transmission - poopunjavanje srednom vrednoscu
    data['transmission'] = data['transmission'].str.extract(r'(\d+)', expand=False).astype(float).fillna(0)
    data["transmission"] = data["transmission"].astype(int)
    suma = data["transmission"].sum()
    brojPodatak = data["transmission"].count()
    data.transmission.replace(0, int(suma/brojPodatak)+1,inplace = True)

    # accident
    data.accident.fillna("None reported", inplace = True)
    data['accident'] = data['accident'].map({'None reported': int(0), 'At least 1 accident or damage reported': int(1)})
    
    # int color - popunjavamo vrednost '-' sa Black, jer je to najvise zastupna boja
    data["int_col"].replace("–",pd.NA,inplace=True)
    data.int_col.fillna("Black", inplace = True)

    # ext color - imamo '-' vrednosti
    data["ext_col"].replace("–",pd.NA,inplace=True)
    data.ext_col.fillna("Black", inplace = True)

    # price - zavisna promenljiva
    data["price"] = data['price'].replace('[\$,]', '', regex=True)

    if ispisVrednosti:
        horsePowerKolona(data)
        enginePowerKolona(data)
        transmissionKolona(data)
        accidentKolona(data)
        intColorKolona(data)
        extColorKolona(data)
        priceKolona(data)

    data["price"] = data["price"].astype(float)

    print(data.info())
    #print(proveriNoneVrednosti(data))
    data = remove_outliers(data)
    return remove_outliers(data)
     


def remove_outliers(df):
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    
    df = df[(df['price'] < upper_bound) & (df['price'] > lower_bound)]
    return df


def visualize_column(df, col_name,  df_fixed=None):
    x = df.index
    plt.plot(x, df[col_name], 'bo', alpha=.2, label='original')
    if df_fixed is not None: plt.plot(x, df_fixed[col_name], 'r-', label='fixed')
    plt.legend()
    plt.show()

def get_model_dropped_col(model, df: pd.DataFrame, get_model_fit, get_error, x, y):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    smallest_error = get_error(model, x_val, y_val)
    best_model = model
    dropped = []
    column_names = x_train.columns.tolist()
    for column in column_names:
        new_df = df.drop(columns=[column])
        new_df = pd.get_dummies(new_df)
        model = get_model_fit(new_df, x_train, y_train)
        x_train_new, x_val_new, y_train_new, y_val_new = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        curr_error = get_error(model, x_val_new, y_val_new)
        if curr_error > smallest_error:
            smallest_error = curr_error
            best_model = model
            dropped = [column]

    '''for i in range(len(column_names)):
        for j in range(len(column_names)):
            if j >= i:
                continue
            column_1 = column_names[i]
            column_2 = column_names[j]
            new_df = df.drop(columns=[column_1, column_2])
            model = get_model_fit(new_df)
            x_train_new, x_val_new, y_train_new, y_val_new = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
            curr_error = get_error(model, x_val_new, y_val_new)
            if curr_error < smallest_error:
                smallest_error = curr_error
                best_model = model
                dropped = [column_1, column_2]'''

    return best_model, dropped