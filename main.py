from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import joblib

app = Flask(__name__)

data = pd.read_csv('procurement_dev.csv')


data['material_name'] = data['material_name'].fillna('')
data['service_name'] = data['service_name'].fillna('')
data['material_cost'] = data['material_cost'].fillna(0)
data['service_cost'] = data['service_cost'].fillna(0)

le_vendor = LabelEncoder()
data['vendor_name_encoded'] = le_vendor.fit_transform(data['vendor_name'])

data['cost'] = data['material_cost'] + data['service_cost']

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

X_train = train_data[['vendor_name_encoded', 'cost']]
X_test = test_data[['vendor_name_encoded', 'cost']]

knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn.fit(X_train)

def recommend_material_ml(material_name, data, model):
    material_data = data[data['material_name'] == material_name]
    if material_data.empty:
        return "No data found for the given material."

    X = material_data[['vendor_name_encoded', 'cost']]
    distances, indices = model.kneighbors(X)

    knn_indices = indices.flatten()
    knn_indices = [i for i in knn_indices if i < len(data)]

    recommendations = data.iloc[knn_indices]
    recommendations = pd.concat([material_data, recommendations])

    recommendations = recommendations[['vendor_name', 'vendor_phone', 'material_cost']].drop_duplicates().sort_values(by='material_cost').head(5)

    output = output = [
    {
        "Vendor name": row['vendor_name'],
        "Vendor phone number": row['vendor_phone'],
        "Material cost": row['material_cost']
    }
    for index, row in recommendations.iterrows()
]
    return output

def recommend_service_ml(service_name, data, model):
    service_data = data[data['service_name'] == service_name]
    if service_data.empty:
        return "No data found for the given service."

    X = service_data[['vendor_name_encoded', 'cost']]
    distances, indices = model.kneighbors(X)

    knn_indices = indices.flatten()
    knn_indices = [i for i in knn_indices if i < len(data)]

    recommendations = data.iloc[knn_indices]
    recommendations = pd.concat([service_data, recommendations])

    recommendations = recommendations[['vendor_name', 'vendor_phone', 'service_cost']].drop_duplicates().sort_values(by='service_cost').head(5)

    output = [
    {
        "Vendor name": row['vendor_name'],
        "Vendor phone number": row['vendor_phone'],
        "Service cost": row['service_cost']
    }
    for index, row in recommendations.iterrows()
]
    return output

@app.route('/recommend/material', methods=['POST'])
def recommend_material():
    data = request.get_json()
    material_name = data['material_name']
    result = recommend_material_ml(material_name, test_data, knn)

    print(result)
    return jsonify(result)

@app.route('/recommend/service', methods=['POST'])
def recommend_service():
    data = request.get_json()
    service_name = data['service_name']
    result = recommend_service_ml(service_name, test_data, knn)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
