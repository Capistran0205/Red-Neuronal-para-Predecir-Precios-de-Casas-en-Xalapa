"""
APLICACIÓN DE PREDICCIÓN DE PRECIOS DE CASAS - XALAPA
=====================================================
Interfaz gráfica moderna con CustomTkinter para predecir precios de casas usando el modelo previamente entrenado.
- Carga el modelo
- Ingreso de las características para calcular el precio de una casa

Fuente: Apoyo de Gemini

Autores: Capistran Ortiz Diego y Baizabal Acosta Ismael
Fecha: 20 de febreo 2026
"""

import customtkinter as ctk
from tkinter import messagebox
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import os


# ══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE APARIENCIA
# ══════════════════════════════════════════════════════════════════
ctk.set_appearance_mode("dark")  
ctk.set_default_color_theme("blue")  


# ══════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL DE LA APLICACIÓN
# ══════════════════════════════════════════════════════════════════
class AppPrediccionCasas(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuración de la ventana
        self.title("🏠 Predicción de Precios de Casas - Xalapa")
        self.geometry("800x750")
        self.resizable(False, False)

        # Cargar modelo y preprocesadores
        self.cargar_modelo()

        # Crear interfaz
        self.crear_widgets()

    def cargar_modelo(self):
        """Carga el modelo entrenado y configura los preprocesadores."""
        try:
            modelo_path = 'modelo_casas_xalapa.keras' 
            if not os.path.exists(modelo_path):
                modelo_path = 'Modelos/modelo_casas_xalapa.keras'
            
            self.modelo = tf.keras.models.load_model(modelo_path)
            print(f"✔ Modelo cargado desde: {modelo_path}")

            df = pd.read_csv('Casas.csv')
            df['prices'] = (
                df['precio']
                .astype(str)
                .str.replace(r'[$,\s]', '', regex=True)
                .astype(float)
            )

            mapping = {
                'tamaño_terreno _m2': 'land_size',
                'tamaño_construccion_m2': 'construction_size',
                'num_recamaras': 'number_rooms',
                'num_baños': 'number_bathrooms',
                'patio': 'garden',
                'roof_garden': 'roof_gardens',
                'num_estacionamientos': 'parking_numbers'
            }
            df.rename(columns=mapping, inplace=True)

            numerical_features = [
                'land_size', 'construction_size', 'number_rooms',
                'number_bathrooms', 'parking_numbers'
            ]
            binary_features = ['garden', 'roof_gardens']

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('bin', 'passthrough', binary_features)
                ],
                verbose_feature_names_out=False
            )

            X = df[numerical_features + binary_features]
            self.preprocessor.fit(X)

            from sklearn.preprocessing import RobustScaler
            self.y_scaler = RobustScaler()
            y = df['prices'].values.reshape(-1, 1)
            self.y_scaler.fit(y)

            print("✔ Preprocesadores configurados correctamente")

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{e}")
            self.destroy()

    def crear_widgets(self):
        """Crea todos los elementos de la interfaz."""
        
        # ──────────────────────────────────────────────────────────
        # TÍTULO
        # ──────────────────────────────────────────────────────────
        titulo = ctk.CTkLabel(
            self,
            text="🏠 Predicción de Precios de Casas",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        titulo.pack(pady=20)

        subtitulo = ctk.CTkLabel(
            self,
            text="Ingresa las características de la propiedad",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        subtitulo.pack(pady=(0, 20))

        # ──────────────────────────────────────────────────────────
        # FRAME PRINCIPAL (CAMPOS DE ENTRADA)
        # ──────────────────────────────────────────────────────────
        frame_principal = ctk.CTkFrame(self)
        frame_principal.pack(padx=40, pady=10, fill="both", expand=True)

        frame_principal.grid_columnconfigure((0, 1), weight=1)

        # ────────── COLUMNA IZQUIERDA ──────────
        ctk.CTkLabel(
            frame_principal,
            text="📐 Tamaño del terreno (m²):",
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")

        self.entry_terreno = ctk.CTkEntry(frame_principal, placeholder_text="Ej: 150", width=250)
        self.entry_terreno.grid(row=1, column=0, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(
            frame_principal,
            text="🏗️ Tamaño de construcción (m²):",
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=2, column=0, padx=20, pady=(0, 5), sticky="w")

        self.entry_construccion = ctk.CTkEntry(frame_principal, placeholder_text="Ej: 120", width=250)
        self.entry_construccion.grid(row=3, column=0, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(
            frame_principal,
            text="🛏️ Número de recámaras:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=4, column=0, padx=20, pady=(0, 5), sticky="w")

        self.entry_recamaras = ctk.CTkEntry(frame_principal, placeholder_text="Ej: 3", width=250)
        self.entry_recamaras.grid(row=5, column=0, padx=20, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(
            frame_principal,
            text="🚿 Número de baños:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=6, column=0, padx=20, pady=(0, 5), sticky="w")

        self.entry_banos = ctk.CTkEntry(frame_principal, placeholder_text="Ej: 2", width=250)
        self.entry_banos.grid(row=7, column=0, padx=20, pady=(0, 20), sticky="ew")

        # ────────── COLUMNA DERECHA ──────────
        ctk.CTkLabel(
            frame_principal,
            text="🚗 Número de estacionamientos:",
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=1, padx=20, pady=(20, 5), sticky="w")

        self.entry_estacionamientos = ctk.CTkEntry(frame_principal, placeholder_text="Ej: 2", width=250)
        self.entry_estacionamientos.grid(row=1, column=1, padx=20, pady=(0, 15), sticky="ew")

        self.check_patio_var = ctk.BooleanVar(value=False)
        self.check_patio = ctk.CTkCheckBox(
            frame_principal,
            text="🌳 Tiene patio",
            variable=self.check_patio_var,
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.check_patio.grid(row=2, column=1, padx=20, pady=(15, 10), sticky="w")

        self.check_roof_var = ctk.BooleanVar(value=False)
        self.check_roof = ctk.CTkCheckBox(
            frame_principal,
            text="🏡 Tiene roof garden",
            variable=self.check_roof_var,
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.check_roof.grid(row=3, column=1, padx=20, pady=(0, 15), sticky="w")

        # ──────────────────────────────────────────────────────────
        # FRAME DE BOTONES (PREDECIR Y LIMPIAR)
        # ──────────────────────────────────────────────────────────
        frame_botones = ctk.CTkFrame(self, fg_color="transparent")
        frame_botones.pack(padx=40, pady=10, fill="x")

        # Botón Limpiar (Gris oscuro/Rojo)
        self.btn_limpiar = ctk.CTkButton(
            frame_botones,
            text="🧹 LIMPIAR",
            command=self.limpiar_campos,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=45,
            width=150,
            fg_color="#4a4a4a",
            hover_color="#c93434"
        )
        self.btn_limpiar.pack(side="left", padx=(0, 10))

        # Botón Predecir (Azul principal)
        self.btn_predecir = ctk.CTkButton(
            frame_botones,
            text="💰 PREDECIR PRECIO",
            command=self.predecir_precio,
            font=ctk.CTkFont(size=16, weight="bold"),
            height=45,
            fg_color="#1f6aa5",
            hover_color="#144870"
        )
        self.btn_predecir.pack(side="right", fill="x", expand=True)

        # ──────────────────────────────────────────────────────────
        # RESULTADO (AHORA CON SCROLL)
        # ──────────────────────────────────────────────────────────
        # Cambiamos CTkFrame por CTkScrollableFrame
        self.frame_resultado = ctk.CTkScrollableFrame(
            self, 
            fg_color="#2b2b2b", 
            height=120,
            orientation="vertical"
        )
        self.frame_resultado.pack(padx=40, pady=(0, 20), fill="both", expand=True)

        self.label_resultado = ctk.CTkLabel(
            self.frame_resultado,
            text="Ingresa los datos y presiona el botón",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        self.label_resultado.pack(pady=25)

    def limpiar_campos(self):
        """Borra el contenido de todos los campos de entrada y resetea el resultado."""
        # Limpiar Entradas de texto
        self.entry_terreno.delete(0, 'end')
        self.entry_construccion.delete(0, 'end')
        self.entry_recamaras.delete(0, 'end')
        self.entry_banos.delete(0, 'end')
        self.entry_estacionamientos.delete(0, 'end')

        # Desmarcar Checkboxes
        self.check_patio_var.set(False)
        self.check_roof_var.set(False)

        # Restaurar etiqueta de resultado
        self.label_resultado.configure(
            text="Ingresa los datos y presiona el botón",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        
        # Devolver el foco al primer campo
        self.entry_terreno.focus()

    def predecir_precio(self):
        """Realiza la predicción del precio."""
        try:
            terreno = float(self.entry_terreno.get())
            construccion = float(self.entry_construccion.get())
            recamaras = int(self.entry_recamaras.get())
            banos = int(self.entry_banos.get())
            estacionamientos = int(self.entry_estacionamientos.get())
            patio = 1 if self.check_patio_var.get() else 0
            roof = 1 if self.check_roof_var.get() else 0

            if terreno <= 0 or construccion <= 0:
                messagebox.showwarning("Advertencia", "Los tamaños deben ser mayores a 0")
                return
            if recamaras < 1 or banos < 1:
                messagebox.showwarning("Advertencia", "Debe haber al menos 1 recámara y 1 baño")
                return
            if estacionamientos < 0:
                messagebox.showwarning("Advertencia", "Los estacionamientos no pueden ser negativos")
                return

            input_data = pd.DataFrame({
                'land_size': [terreno],
                'construction_size': [construccion],
                'number_rooms': [recamaras],
                'number_bathrooms': [banos],
                'parking_numbers': [estacionamientos],
                'garden': [patio],
                'roof_gardens': [roof]
            })

            X_preprocessed = self.preprocessor.transform(input_data)
            prediccion_normalizada = self.modelo.predict(X_preprocessed, verbose=0)
            precio_predicho = self.y_scaler.inverse_transform(prediccion_normalizada)[0][0]

            precio_min = precio_predicho * 0.90
            precio_max = precio_predicho * 1.10

            # Como ahora es un ScrollableFrame, podemos agregar más detalles si queremos
            self.label_resultado.configure(
                text=f"💰 PRECIO ESTIMADO:\n\n"
                     f"${precio_predicho:,.0f} MXN\n\n"
                     f"Rango de confianza (±10%):\n"
                     f"${precio_min:,.0f} - ${precio_max:,.0f}\n\n"
                     f"───────────────────────────────\n"
                     f"Características analizadas:\n"
                     f"Terreno: {terreno}m² | Construcción: {construccion}m²\n"
                     f"Recámaras: {recamaras} | Baños: {banos}\n"
                     f"Estacionamientos: {estacionamientos}\n",
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color="#4CAF50"
            )

        except ValueError:
            messagebox.showerror("Error", "Por favor verifica que todos los campos numéricos sean válidos.")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error:\n{e}")

# ══════════════════════════════════════════════════════════════════
# EJECUTAR APLICACIÓN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = AppPrediccionCasas()
    app.mainloop()