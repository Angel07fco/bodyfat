from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('bodyfat_modelo.pkl')
scaler = joblib.load('bodyfat_escalado.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        density = float(request.form['density'])
        age = float(request.form['age'])
        chest =float(request.form['chest'])
        abdomen =float(request.form['abdomen'])
        biceps =float(request.form['biceps'])

        input_data = pd.DataFrame({
            'Density': [density],
            'Age': [age],
            'Weight': [0],
            'Height': [0],
            'Neck': [0],
            'Chest': [chest],
            'Abdomen': [abdomen],
            'Hip': [0],
            'Thigh': [0],
            'Knee': [0],
            'Ankle': [0],
            'Biceps': [biceps],
            'Forearm': [0],
            'Wrist': [0]
        })

        # Escalar los datos de entrada
        scaled_data = scaler.transform(input_data)

        # Seleccionar solo las características usadas para el modelo
        scaled_data_for_prediction = scaled_data[:, [0,1,5,6,11]]  # Asegúrate de que estos índices son correctos

        # Realizar la predicción con los datos escalados
        prediccion = model.predict(scaled_data_for_prediction)

        # Devolver la predicción como JSON
        prediction_value = round(float(prediccion[0]), 2)

        return jsonify({'prediction': prediction_value})

    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

