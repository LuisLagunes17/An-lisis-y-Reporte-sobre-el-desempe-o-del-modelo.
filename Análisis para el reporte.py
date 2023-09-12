# Importamos las librerías que se usaran para este modelo
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


# Seleccionamos nuestros datos con los cuales vamos a trabajar, este es un dataf
# me de un banco con caracteristicas importantes
# de sus clientes como edad, profesión, crédito, estatus social, entre otras.
data = 'dataset.csv'

try:
    dataset = pd.read_csv("C:/Users/luisl/Documents/Módulo de Uresti/dataset.csv")
    # Realiza operaciones con el DataFrame 'df'
except FileNotFoundError:
    print(f"El archivo 'C:/Users/luisl/Documents/Módulo de Uresti/dataset.csv' no fue encontrado.")

# En este paso vamos a eliminar los datos faltantes, en nuestro dataset los datos faltantes estan expresados por 'unknown', para ello los remplazamos
# por NA para después aplicar una función que borre todas las filas que tengan algún dato faltante en alguna de nuestras columnas, esto con el fin
# de poder tener nuestra base de datos limpia para un mejor modelo
dataset.replace('unknown', pd.NA, inplace=True)
data_limpia = dataset[dataset != '<NA>'].dropna()

# Tenemos la variable edad la cual esta representada por varias edad por lo cuál se procedio a categorizarla por rango de edades, 
# en este caso por 5 categorías
intervalos = [0, 18, 40, 60, 80, 100]
labels = ['0-18', '19-40', '41-60', '61-80','81-100']
data_limpia['age'] = pd.cut(data_limpia['age'], bins=intervalos, labels=labels, right=False)

# Se procedio a utilizar la cantidad de datos correspondientes para que no existiera un sezgo en los datos por la cantidad de datos, es decir que el modelo
# no entre solo un tipo de respuesta o valor, es decir buscamos que el modelo escoja datos variados y no caiga en un sezgo
si_samples = data_limpia[data_limpia['housing'] == 'yes'].sample(n=2000)
no_samples = data_limpia[data_limpia['housing'] == 'no'].sample(n=2000)

# Concadenamos los valores seleccionados
data_limpia = pd.concat([si_samples, no_samples])
# Déspues ponemos los datos aleatorios para que no esten ordenados primeros lo de "yes" y "no"
data_limpia = data_limpia.sample(frac=1, random_state=42)

# Crear una instancia de LabelEncoder
label_encoder = LabelEncoder()

# Aplicar Label Encoding a cada columna categórica esto debido a que nuestro modelo solo recibe valores numéricos
columns_to_encode = ['age', 'job', 'marital', 'housing', 'contact', 'month', 'day_of_week', 'previous', 'education']
for column in columns_to_encode:
    data_limpia[column] = label_encoder.fit_transform(data_limpia[column])
    
# Aquí seleccionamos nuestras variables tanto y como X, en este caso nuestra y va ser "loan" y nuestras variables X son las variables más significativas y que
# pueden influir de manera significa en nuestro modelo y no solo empeore nuestro modelo, en este caso son 8 variables.
Y = data_limpia['loan'].values 
X = data_limpia[['age', 'job', 'marital', 'housing', 'contact',
                         'month', 'day_of_week',
                         'previous']].values 

# Aquí almacenamos nuestro valores de las métricas para cada iteración para poder después graficar el comportamiento.
accuracy_scores = []
f1_scores = []
recall_scores = []

# Realizamos 100 iteraciones para evaluar los resultados
for i in range(100):
    # Divide los datos en conjuntos de entrenamiento y prueba de forma aleatoria
    # En este caso tenemos el 5% de los datos para evuarlos con el modelo y tenemos la semilla 42 para escojer datos aleatorios de cada uno de nuestros datos
    # que seleccionamos para que muestre el aprendizaje del modelo.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=np.random.randint(42))
    model = BernoulliNB(fit_prior = False)

    # Entrenamos el modelo
    model.fit(X_train, y_train)

    # Realizamos las predicciones en los datos de prueba
    y_pred = model.predict(X_test)

    # Se calculan y almacenan las métricas en cada iteración
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    recall_scores.append(recall)

    # Imprimimos los resultados de métricas en esta iteración
    print(f"Iteración {i + 1}:")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Recall: {recall}")
    print("=" * 40)

# Crea los subplots para cada métrica
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Gráfica de Accuracy
axs[0].plot(range(1, 101), accuracy_scores, color='b')
axs[0].set_title('Accuracy')

# Gráfica de F1 Score
axs[1].plot(range(1, 101), f1_scores, color='g')
axs[1].set_title('F1 Score')

# Gráfica de Recall
axs[2].plot(range(1, 101), recall_scores, color='r')
axs[2].set_title('Recall')

# Ajustar espaciado entre subplots
plt.tight_layout()

# Mostrar las gráficas
plt.show()

# Calcular la varianza de Accuracy, F1 Score y Recall
varianza_accuracy = np.var(accuracy_scores)
varianza_f1 = np.var(f1_scores)
varianza_recall = np.var(recall_scores)

print(f'Varianza de Accuracy: {varianza_accuracy:.4f}')
print(f'Varianza de F1 Score: {varianza_f1:.4f}')
print(f'Varianza de Recall: {varianza_recall:.4f}')

# Aquí almacenamos nuestro valores de las métricas para cada iteración para poder después graficar el comportamiento.
accuracy_scores_2 = []
f1_scores_2 = []
recall_scores_2 = []

# Realizamos 100 iteraciones para evaluar los resultados
for i in range(100):
    # Divide los datos en conjuntos de entrenamiento y prueba de forma aleatoria
    # En este caso tenemos el 5% de los datos para evuarlos con el modelo y tenemos la semilla 42 para escojer datos aleatorios de cada uno de nuestros datos
    # que seleccionamos para que muestre el aprendizaje del modelo.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=np.random.randint(42))

    # Aquí definimos que modelo vamos a ocupar en este caso Naive Bayes Bernoulli ya que este modelo se acopla de manera más significativa a los datos
    # El suavizado fue de 0.5 ya que daba estabilidad al modelo y bajar el sezgo en los datos de manera significativa, utilizamos fit_prior para poder
    # hacer que el modelo aprenda las probabilidades previas de la clase lo que hace que mejore.
    model = BernoulliNB(alpha=0.5, fit_prior = True, class_prior= None)

    # Entrenamos el modelo
    model.fit(X_train, y_train)

    # Realizamos las predicciones en los datos de prueba
    y_pred = model.predict(X_test)

    # Se calculan y almacenan las métricas en cada iteración
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    accuracy_scores_2.append(accuracy)
    f1_scores_2.append(f1)
    recall_scores_2.append(recall)

    # Imprimimos los resultados de métricas en esta iteración
    print(f"Iteración {i + 1}:")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Recall: {recall}")
    print("=" * 40)

# Crea los subplots para cada métrica
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Gráfica de Accuracy
axs[0].plot(range(1, 101), accuracy_scores_2, color='b')
axs[0].set_title('Accuracy')

# Gráfica de F1 Score
axs[1].plot(range(1, 101), f1_scores_2, color='g')
axs[1].set_title('F1 Score')

# Gráfica de Recall
axs[2].plot(range(1, 101), recall_scores_2, color='r')
axs[2].set_title('Recall')

# Ajustar espaciado entre subplots
plt.tight_layout()

# Mostrar las gráficas
plt.show()

# Calcular la varianza de Accuracy, F1 Score y Recall
varianza_accuracy = np.var(accuracy_scores_2)
varianza_f1 = np.var(f1_scores_2)
varianza_recall = np.var(recall_scores_2)

print(f'Varianza de Accuracy: {varianza_accuracy:.4f}')
print(f'Varianza de F1 Score: {varianza_f1:.4f}')
print(f'Varianza de Recall: {varianza_recall:.4f}')

# Crear subplots para comparar métricas entre modelos
fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# Gráficas de Accuracy
axs[0, 0].plot(range(1, 101), accuracy_scores, color='b', label='Modelo 1')
axs[0, 0].plot(range(1, 101), accuracy_scores_2, color='r', label='Modelo 2')
axs[0, 0].set_title('Accuracy')
axs[0, 0].legend()

# Gráficas de F1 Score
axs[1, 0].plot(range(1, 101), f1_scores, color='b', label='Modelo 1')
axs[1, 0].plot(range(1, 101), f1_scores_2, color='r', label='Modelo 2')
axs[1, 0].set_title('F1 Score')
axs[1, 0].legend()

# Gráficas de Recall
axs[2, 0].plot(range(1, 101), recall_scores, color='b', label='Modelo 1')
axs[2, 0].plot(range(1, 101), recall_scores_2, color='r', label='Modelo 2')
axs[2, 0].set_title('Recall')
axs[2, 0].legend()

# Ajustar espaciado entre subplots
plt.tight_layout()

# Mostrar las gráficas
plt.show()