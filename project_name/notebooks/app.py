from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route("/recommend", methods=['POST'])
def recommend():
    # Load data from CSV file
    df = pd.read_csv("RawData.csv")
    
    # Preprocess the 'price' column
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    
    # Define the target price and search range
    precio_objetivo = 50
    rango = 10
    df_f = df[(df['price'] >= precio_objetivo - rango) & (df['price'] <= precio_objetivo + rango)]
    
    # Prepare input and output variables
    X = df_f[['price']].values
    y = df_f['price'].values
    
    # Split the dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the KNN model and train it
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Receive input data (cofie) from the client
    content = request.get_json()
    if not content or 'cofie' not in content:
        return jsonify({'error': 'Missing "cofie" vector in the request'}), 400
    
    # Convert the received 'cofie' into a numpy array and make a prediction
    cofie = np.array(content['cofie']).reshape(1, -1)
    prediction = knn.predict(cofie)
    
    # Respond with the prediction
    return jsonify({"recommended_cofie": prediction.tolist()})

if __name__ == "_main_":
    app.run(debug=True)