import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

#*Librerias para evaluar las clasificaciones
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# Función para cargar y dividir los datos de Swedish Auto Insurance Dataset
def read_dataAutoInsur():
    dataset = pd.read_csv('AutoInsurSweden.csv')
    X = dataset[['X']] #X = number of claims
    y = dataset['Y'] #Y = total payment for all the claims in thousands of Swedish Kronor for geographical zones in Sweden
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Función para cargar y dividir los datos de Wine Quality Dataset
def read_dataWineQuality():
    dataset = pd.read_csv('wine-Quality.csv', sep=",")
    # Separar las características (X) y la variable objetivo (y)
    X = dataset.drop("quality", axis=1)
    y = dataset["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Función para cargar y dividir los datos de Pima Indians Diabetes
def read_datapima_Diabetes():
    dataset = pd.read_csv('pima-indians-diabetes.csv', sep=",")
    X = dataset.drop("Class variable (0 or 1)", axis=1)
    y = dataset["Class variable (0 or 1)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Modelos de regresión
def logistic_Regression(X_train, X_test, y_train, y_test, op):
    model = LogisticRegression(max_iter=10000)
    # Ajustar el modelo a los datos de entrenamiento
    model.fit(X_train, y_train)
    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)
    #* Nuevas metricas, solo algunas se pueden mostrar dependiendo el tipo de problema
    #*Para el archivo 2
    if op == 2:
        accuracyLR = accuracy_score(y_test, y_pred)
        print(f"Logistic Regression Accuracy: {accuracyLR}")
    #* Para el archivo 3
    elif op == 3:
        accuracyLR = accuracy_score(y_test, y_pred)
        precisionLR = precision_score(y_test, y_pred)
        print(f"Logistic Regression Accuracy: {accuracyLR}")
        print(f"Logistic Regression Precision: {precisionLR}")
    

def k_Nearest_Neighbors(X_train, X_test, y_train, y_test, op, n_neighbors=3):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("K-Nearest Neighbors Mean Squared Error:", mse)

def support_Vector_Machine(X_train, X_test, y_train, y_test, op):
    model = SVR()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Support Vector Machine Mean Squared Error:", mse)

def naive_Bayes(X_train, X_test, y_train, y_test, op):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #*Para el archivo 2
    if op == 2:
        accuracyNB = accuracy_score(y_test, y_pred)
        print(f"Naive Bayes Accuracy: {accuracyNB}")
    #* Para el archivo 3
    elif op == 3:
        accuracyNB = accuracy_score(y_test, y_pred)
        precisionNB = precision_score(y_test, y_pred)
        print(f"Naive Bayes Accuracy: {accuracyNB}")
        print(f"Naive Bayes Precision: {precisionNB}")

def MLP(X_train, X_test, y_train, y_test, op, hidden_layer_sizes=(100,50), max_iter=500):
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MLP (Neural Network) Mean Squared Error:", mse)

# Lista de archivos
file_names = ['AutoInsurSweden.csv','wine-Quality.csv', 'pima-indians-diabetes.csv']

while True:
    print("\nSeleccione un dataset:")
    for i, file_name in enumerate(file_names, start=1):
        print(f"{i}. {file_name}")

    option = input("Ingrese el número del dataset que desea utilizar (o 's' para salir): ")

    if option.lower() == 's':
        break

    try:
        option = int(option)
        if 1 <= option <= len(file_names):
            file_name = file_names[option - 1]
            print(f"\nDataset seleccionado: {file_name}")
            # Cargar y dividir datos según el dataset seleccionado
            if file_name == 'AutoInsurSweden.csv':
                X_train, X_test, y_train, y_test = read_dataAutoInsur()
            elif file_name == 'wine-Quality.csv':
                X_train, X_test, y_train, y_test = read_dataWineQuality()
            elif file_name == 'pima-indians-diabetes.csv':
                X_train, X_test, y_train, y_test = read_datapima_Diabetes()
                
            # Aplicar todos los modelos al dataset seleccionado
            logistic_Regression(X_train, X_test, y_train, y_test, option)
            k_Nearest_Neighbors(X_train, X_test, y_train, y_test, option)
            support_Vector_Machine(X_train, X_test, y_train, y_test, option)
            naive_Bayes(X_train, X_test, y_train, y_test, option)
            MLP(X_train, X_test, y_train, y_test, option)

        else:
            print("Número de dataset no válido. Inténtelo de nuevo.")
    except ValueError as e:
        print("Entrada no válida. Ingrese un número válido o 's' para salir.", e)