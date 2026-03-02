"""
PREPROCESAMIENTO DE DATOS PARA REGRESIÃ“N LINEAL MÃšLTIPLE
========================================================
Este script realiza todo el preprocesamiento necesario del dataset:
- Conversión de tipo de datos
- Normalización de los datos
- Divisiónn del conjutno de entrenamiento

Fuente: Apoyo de Gemini

Autores: Capistran Ortiz Diego y Baizabal Acosta Ismael
Fecha: 18 de febrero 2026

"""

# Imports necesarios para el preprocesamiento
import pandas as pd
# Seleccionar el modelo de entrenamiento 
from sklearn.model_selection import train_test_split
# NormalizaciÃ³n Z-Score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class PreprocesadorCasas:
    """
    Clase para preprocesar el dataset de casas, separarlo y normalizarlo correspondientemente.    
    """
    # Constructor 
    def __init__(self, filepath):
        self.filepath = filepath # Ruta del archivo
        self.df_original = None # Conjunto de datos origianales
        self.df_processed = None # Valores de X para Entrenar
        self.X_train = None # Valores de X para Entrenamiento
        self.X_test = None # Valores de X para Test
        self.y_train = None # Valores de y para Entrenamiento
        self.y_test = None # Valores de y para Test
        self.y_scaler = StandardScaler() # Inicialización de la Normalización para uso posterior
        self.min_max = MinMaxScaler() # Inicialización de la Normalización para uso posterior
        self.feature_names = [] # Lista para las características que manejara el modelo
        self.preprocessor = None # Para el transformador de X

    # Función para cargar los datos del archivo .cvs
    def cargar_datos(self):
        # Cargar el dataset esde el archivo CSV
        print("Cargando Datos...")
        try:
            self.df_original = pd.read_csv(self.filepath) # Apertura del archivo
            self.df_original.dropna(how = 'all', inplace  = True) # Reemplazar registros vacios
            self.feature_names = self.df_original.columns.to_list()
            print(f"Dataset cargado")
            print(f" Total de registros obtenidos: {len(self.df_original)}")
            print(f" Total de columnas: {len(self.df_original.columns)}")
        except Exception as e:
            print(f"No se logro abrir el archivo csv: {e}")
    
    # Conversión para los tipos de datos correspondientes del csv
    # * tamaño_terreno (float)
    # * tamaño_construccion (float)
    # * num_recamaras (int)
    # * num_baños (int)
    # * patio (int)
    # * roof_garden (float)
    # * num_estacionamientos (int)
    # * cp (int), pero por códificación (str)
    # * precio (int)
    def conversion_campos(self):
        # Copia del original para procesarlo
        self.df_processed = self.df_original.copy()
        #
        # 1. Limpieza de Precios (Quitar $ y comas)
        self.df_processed['prices'] = (
                self.df_processed['precio']
                .astype(str)
                .str.replace(r'[$,]', '', regex=True)
                .astype(float)
             )
            
        # 2. Convertir CP a string para que sea categÃ³rico
        self.df_processed['postal_codes'] = 'CP_' + self.df_processed['cp'].astype(str)
        
        # 3. Renombrar y asegurar tipos numÃ©ricos para el resto
        mapping = {
            'tamaño_terreno _m2': 'land_size',
            'tamaño_construccion_m2': 'construction_size',
            'num_recamaras': 'number_rooms',
            'num_baños': 'number_bathrooms',
            'patio': 'garden',
            'roof_garden': 'roof_gardens',
            'num_estacionamientos': 'parking_numbers'
        }
        
        # Renombramos columnas si existen en el csv
        self.df_processed.rename(columns=mapping, inplace=True)
        
        # Aseguramos que las columnas renombradas sean float/int
        numeric_cols = list(mapping.values())
        for col in numeric_cols:
            if col in self.df_processed.columns:
                self.df_processed[col] = pd.to_numeric(self.df_processed[col], errors='coerce')
        
        # Eliminar filas con NaN que pudieron generarse
        self.df_processed.dropna(inplace=True)
        print(f" Conversión de tipos finalizada. Registros válidos: {len(self.df_processed)}")
    # Función para normalizar el conjunto de datos
    def normalizar_datos(self):
            # Definición de columnas
            target_col = 'prices'
            
            # Columnas numéricas (StandardScaler)
            numerical_features = ['land_size', 'construction_size', 'number_rooms', 'number_bathrooms', 'parking_numbers']
            
            # Columnas binarias (Passthrough)
            binary_features = ['garden', 'roof_gardens']
            
            # === 1. Preprocesamiento de X (Features) ===
            # Eliminamos el OneHotEncoder ya que descartaremos el CP
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('bin', 'passthrough', binary_features)
                ],
                verbose_feature_names_out=False
            )
            
            # Separar X e y (excluyendo 'postal_codes')
            X = self.df_processed[numerical_features + binary_features]
            y = self.df_processed[target_col].values.reshape(-1, 1)
            
            # Ajustar transformadores
            X_normalized = self.preprocessor.fit_transform(X)
            y_normalized = self.y_scaler.fit_transform(y)
            
            # === 2. División de datos para Train / Test ===
            # Uso del 20% para test (aprox 18 registros de los 90)
            test_size = 18 
            train_size = len(X_normalized) - test_size
            
            self.X_train = X_normalized[:train_size]
            self.X_test = X_normalized[train_size:]
            
            self.y_train = y_normalized[:train_size]
            self.y_test = y_normalized[train_size:]
                
            print(f" Normalización completada.")
            print(f"     - Train set: {self.X_train.shape}")
            print(f"     - Test set:  {self.X_test.shape}")
    
    # Función para ejecutar secuencialmente las anteriores funciones
    def execute_preprocesador(self):
        print("\n")
        print("Ejecución del preprocesamiento de datos de entrenamiento")
        print("\n")

        # Ejecutar todas las funciones previas
        self.cargar_datos()
        self.conversion_campos()
        self.normalizar_datos()
        
        print("\n")
        print(" PREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
        print(f"\nDatos listos para entrenamiento:")
        print(f"  - Shape X_train: {self.X_train.shape}")
        print(f"  - Shape X_test: {self.X_test.shape}")
        print(f"  - Shape y_train: {self.y_train.shape}")
        print(f"  - Shape y_test: {self.y_test.shape}")
        print(f"  - Features: {len(self.feature_names)}")
        return self.X_train, self.X_test, self.y_train, self.y_test, self.y_scaler