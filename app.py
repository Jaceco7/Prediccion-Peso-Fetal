from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "clave_secreta"

# CARGAR MODELOS 
modelo_logistica = joblib.load("modelos/modelo_logistica.pkl")
modelo_mlp = joblib.load("modelos/modelo_mlp.pkl")
modelo_svm = joblib.load("modelos/modelo_svm.pkl")
escalador_fcm = joblib.load("modelos/escalador_fcm.pkl")
pesos_fcm = joblib.load("modelos/pesos_fcm.pkl")
escalador_mlp = joblib.load("modelos/escalador_mlp.pkl")

#  VARIABLES DE ENTRADA EN ORDEN CORRECTO 
columnas_modelo = [f"C{i}" for i in range(1, 31)]  

# FUNCIÓN FCM
def predecir_fcm(X, pesos):
    activacion = np.dot(X, pesos)
    return (activacion > 0.5).astype(int)

# RUTAS WEB 
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/form_individual')
def form_individual():
    return render_template("form_individual.html")

@app.route('/form_lote')
def form_lote():
    return render_template("form_lote.html")

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        
        datos = [float(request.form[f"input{i}"]) for i in range(1, 31)]
        modelo_seleccionado = request.form["modelo"]

        
        df = pd.DataFrame([datos], columns=columnas_modelo)

        
        df_escalado = escalador_fcm.transform(df)

        resultado = None

        if modelo_seleccionado == "logistica":
            pred = modelo_logistica.predict(df_escalado)[0]
            resultado = ("Regresión Logística", pred)
        elif modelo_seleccionado == "mlp":
            pred = modelo_mlp.predict(df_escalado)[0]
            resultado = ("Red Neuronal (MLP)", pred)
        elif modelo_seleccionado == "svm":
            pred = modelo_svm.predict(df_escalado)[0]
            resultado = ("Máquina SVM", pred)
        elif modelo_seleccionado == "fcm":
            pred = predecir_fcm(df_escalado, pesos_fcm)[0]
            resultado = ("Mapa Cognitivo Difuso (FCM)", pred)

        return render_template("resultado_individual.html", resultado=resultado[1], modelo_usado=resultado[0])

    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/lote', methods=['POST'])
def lote():
    try:
        archivo = request.files['archivo']
        modelo_seleccionado = request.form["modelo"]

        
        if not archivo.filename.endswith(('.xlsx', '.xls', '.csv')):
            return render_template("error.html")

        
        if archivo.filename.endswith('.csv'):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo)

        X = df[columnas_modelo]
        y = df['C31']

        resultados = {}
        matrices = {}

        for nombre, modelo, clave in [
            ("Logística", modelo_logistica, "logistica"),
            ("MLP", modelo_mlp, "mlp"),
            ("SVM", modelo_svm, "svm")
        ]:
            pred = modelo.predict(X)
            acc = accuracy_score(y, pred)
            cm = confusion_matrix(y, pred)
            plt.figure()
            sns.heatmap(cm, annot=True, fmt='d')
            plt.title(f"Matriz de Confusión - {nombre}")
            path_img = f"static/cm_{clave}.png"
            plt.savefig(path_img)
            resultados[nombre] = acc
            matrices[nombre] = path_img

        X_fcm = escalador_fcm.transform(X)
        pred_fcm = predecir_fcm(X_fcm, pesos_fcm)
        acc_fcm = accuracy_score(y, pred_fcm)
        cm_fcm = confusion_matrix(y, pred_fcm)
        plt.figure()
        sns.heatmap(cm_fcm, annot=True, fmt='d')
        plt.title("Matriz de Confusión - FCM")
        path_fcm = "static/cm_fcm.png"
        plt.savefig(path_fcm)
        resultados["FCM"] = acc_fcm
        matrices["FCM"] = path_fcm

        modelo_nombre_map = {
            "logistica": "Logística",
            "mlp": "MLP",
            "svm": "SVM",
            "fcm": "FCM"
        }

        modelo_usuario = modelo_nombre_map[modelo_seleccionado]
        exactitud = resultados[modelo_usuario]

        return render_template(
            "resultado_lote.html",
            resultados=resultados,
            matrices=matrices,
            modelo_usuario=modelo_usuario,
            modelo_usado=modelo_seleccionado,
            exactitud=exactitud
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
