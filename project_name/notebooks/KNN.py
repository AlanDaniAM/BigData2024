import numpy as np
from collections import Counter
from scipy.spatial import distance
from flask import Flask, request, jsonify
from collections import Counter
import numpy as np

class KNearestNeighbors:
    def _init_(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        # Entrenar el modelo guardando los datos de entrenamiento
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        # Realizar predicciones para el conjunto de prueba
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
    
    def _predict(self, x):
        # Calcular distancias euclidianas entre x y cada muestra en X_train
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # Seleccionar los k vecinos más cercanos
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Determinar la etiqueta más común entre los vecinos
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

app = Flask(__name__)

class KNN:
    def _init_(self, data, k=1, metric="euclidean"):
        self.k = k
        self.metric = metric
        self.data = data

    # Metodo para entrenar
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # Metodo para hacer recomendaciones
    def predict(self, VG):
        distances = []
        for i, row in enumerate(self.data):
            dist = distance.euclidean(row, VG)
            distances.append((dist, i))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]
        return [self.data[i[1]] for i in neighbors]

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        VG = request.get_json()['VG']
        if not VG:
            return jsonify({'error': 'Game is required'}), 400
        knn = KNN([[1, 2], [3, 4], [5, 6]], k=3)
        knn.fit([[1, 2], [3, 4], [5, 6]], [1, 2, 3])
        recommended_VG = knn.predict(VG)
        return jsonify({'recommended_VG': recommended_VG})
    except KeyError:
        return jsonify({'error': 'Invalid request'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

class KNearestNeighbors:
    def _init_(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]