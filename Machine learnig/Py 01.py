import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Carga del dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Exploración inicial
print(df.head())  # Primeras filas
print(df.info())  # Información general
print(df.describe())  # Estadísticas básicas

# 2. Preprocesamiento de datos
# Manejo de valores nulos
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Codificación de variables categóricas
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])

# Selección de características y variable objetivo
X = df[['Pclass', 'Age', 'Fare', 'Sex']]
y = df['Survived']

# Normalización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Implementación de un modelo de Machine Learning
modelo = LogisticRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')

# 4. Operaciones de Álgebra Lineal en Python
vector = np.array([2, 3, 4])
matriz = np.array([[1, 2], [3, 4]])
producto_punto = np.dot(vector, vector)
norma = np.linalg.norm(vector)

# Resolución de un sistema de ecuaciones lineales
A = np.array([[3, 2], [4, 1]])
b = np.array([5, 6])
solucion = np.linalg.solve(A, b)
print("Producto punto:", producto_punto)
print("Norma del vector:", norma)
print("Solución del sistema de ecuaciones:", solucion)

# 5. Análisis Estadístico de Datos
print("Media de la edad:", df['Age'].mean())
print("Mediana de la edad:", df['Age'].median())

# Visualización de datos
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Distribución de Edad")
plt.show()

# 6. Identificación de valores atípicos
std = df['Age'].std()
mean = df['Age'].mean()
outliers_std = df[(df['Age'] < mean - 3 * std) | (df['Age'] > mean + 3 * std)]
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df[(df['Age'] < Q1 - 1.5 * IQR) | (df['Age'] > Q3 + 1.5 * IQR)]
print("Valores atípicos detectados por desviación estándar:", outliers_std.shape[0])
print("Valores atípicos detectados por IQR:", outliers_iqr.shape[0])

# 7. Implementación de funciones estadísticas personalizadas
def calcular_varianza(datos):
    media = sum(datos) / len(datos)
    return sum((x - media) ** 2 for x in datos) / len(datos)

def calcular_desviacion_estandar(datos):
    return calcular_varianza(datos) ** 0.5

# Aplicación en los datos
edades = df['Age'].dropna().tolist()
print("Varianza manual:", calcular_varianza(edades))
print("Desviación estándar manual:", calcular_desviacion_estandar(edades))
print("Varianza con NumPy:", np.var(edades))
print("Desviación estándar con NumPy:", np.std(edades))
