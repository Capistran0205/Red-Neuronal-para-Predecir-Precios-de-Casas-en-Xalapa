"""
CREACIÓN, ENTRENAMIENTO, PRUEBA  Y GUARDADO DE UN MODELO DE RED NEUORNAL
========================================================
Este script realiza la creación del modelo, su entrenamiento, prueba y guardado:
- Crea el modelo
- Entrena el modelo
- Prueba del modelo y Guardado de este mismo.

Fuente: Apoyo de Gemini

Autores: Capistran Ortiz Diego y Baizabal Acosta Ismael
Fecha: 20 de febrero 2026

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from DataSetReader import PreprocesadorCasas # Importamos tu clase
# Función para crear el modelo de red neuoronal como múltiples capas
def crear_modelo(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=[input_shape]), # Capa de entrada con 64 neuronas y función relu
        Dense(32, activation='relu'), # Capa Oculta con 32 neuronas y función relu
        Dense(16, activation='relu'), # Capa Oculta con 16 neuronas y función relu
        Dense(1, activation=None) # Capa de Salida con 1 neurona y por defecto es linear para predecir el precio.
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Función principal para la ejcución del modelo de red neuronal
def main():
    # 1. Instanciar y ejecutar tu preprocesador
    lector = PreprocesadorCasas('Casas.csv')
    X_train, X_test, y_train, y_test, y_scaler = lector.execute_preprocesador()
    
    def entrenar_nuevo_modelo():
        print("\nEntrenando la red neuronal...")
        modelo = crear_modelo(X_train.shape[1])
        modelo.fit(X_train, y_train, epochs=1000, validation_split=0.2, verbose=1)
        
        # Evaluar
        loss, mae = modelo.evaluate(X_test, y_test, verbose=0)
        mae_real = y_scaler.inverse_transform([[mae]])[0][0] - y_scaler.inverse_transform([[0]])[0][0]
        print(f"Entrenamiento completado. Error absoluto medio (Test): +/- ${abs(mae_real):,.2f} MXN")
        return modelo
    
    modelo = entrenar_nuevo_modelo() # Entrenamiento inicial
    
    # 2. Menú interactivo
    while True:
        print("\n--- PREDICCIÓN DE PRECIO DE CASA ---")
        try:
            t_terreno = float(input("Tamaño del terreno (m2): "))
            t_construccion = float(input("Tamaño de la construcción (m2): "))
            n_recamaras = int(input("Número de recámaras: "))
            n_banos = int(input("Número de baños: "))
            patio = int(input("¿Tiene patio? (1 = Sí, 0 = No): "))
            roof = int(input("¿Tiene roof garden? (1 = Sí, 0 = No): "))
            n_estac = int(input("Número de estacionamientos: "))
            
            # Crear DataFrame con los nombres en inglés que se definió en el maping()
            nueva_casa = pd.DataFrame([{
                'land_size': t_terreno,
                'construction_size': t_construccion,
                'number_rooms': n_recamaras,
                'number_bathrooms': n_banos,
                'garden': patio,
                'roof_gardens': roof,
                'parking_numbers': n_estac
            }])
            
            # Normalizar usando tu ColumnTransformer
            nueva_casa_scaled = lector.preprocessor.transform(nueva_casa)
            
            # Predecir
            precio_pred_scaled = modelo.predict(nueva_casa_scaled, verbose=0)
            precio_pred = y_scaler.inverse_transform(precio_pred_scaled)[0][0]
            
            print(f"\n======================================")
            print(f"PRECIO ESTIMADO POR LA RED: ${precio_pred:,.2f} MXN")
            print(f"======================================")
            
            # 3. Gráfico de dispersión (Tamaño de Construcción vs Precio)
            plt.figure(figsize=(10, 6))
            
            # Extraemos los datos reales sin normalizar desde el dataframe procesado
            df_plot = lector.df_processed
            plt.scatter(df_plot['construction_size'], df_plot['prices'], color='blue', alpha=0.6, label='Datos Reales de Entrenamiento')
            
            # Graficamos la predicción actual
            plt.scatter(t_construccion, precio_pred, color='red', marker='*', s=350, edgecolor='black', label='Tu Predicción Actual')
            
            plt.title('Dispersión: Metros de Construcción vs Precio Estimado')
            plt.xlabel('Tamaño de Construcción (m²)')
            plt.ylabel('Precio en Millones ($MXN)')
            plt.legend()
            plt.grid(True)
            plt.show()
            
            print("\n¿Qué deseas hacer ahora?")
            print("1. Guardar modelo (.keras)")
            print("2. Reentrenar el modelo")
            print("3. Predecir otra casa")
            print("4. Salir")
            opcion = input("Elige una opción (1-4): ")
            
            if opcion == '1':
                modelo.save('Modelos/modelo_casas_xalapa.keras')
                print("¡Modelo guardado!")
                break
            elif opcion == '2':
                modelo = entrenar_nuevo_modelo()
            elif opcion == '3':
                continue
            elif opcion == '4':
                break
            else:
                print("Opción inválida.")
                
        except ValueError:
            print("Por favor, ingresa solo números válidos.")

if __name__ == '__main__':
    main()