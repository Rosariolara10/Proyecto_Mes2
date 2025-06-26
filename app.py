from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Cargar modelo previamente entrenado
modelo = joblib.load("modelo_regresion.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    if request.method == 'POST':
        try:
            # Obtener valores del formulario
            valores = [
                float(request.form['num_app_switches']),
                float(request.form['sleep_hours']),
                float(request.form['notification_count']),
                float(request.form['social_media_time_min']),
                float(request.form['focus_score']),
                float(request.form['anxiety_level']),
            ]
            entrada = np.array([valores])
            prediccion = modelo.predict(entrada)
            resultado = round(prediccion[0], 2)
        except Exception as e:
            resultado = f"❌ Error: {e}"
    return render_template('index.html', resultado=resultado)

if __name__ == '__main__':
    app.run(debug=True)
