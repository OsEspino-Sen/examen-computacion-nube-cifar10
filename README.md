# Examen Final — Computación en la Nube | UTH

## CIFAR Vision

Aplicación web de clasificación de imágenes utilizando una **Red Neuronal Convolucional (CNN)** entrenada con **CIFAR-10** en Google Colab y desplegada mediante **Streamlit**.

## 🌐 URL DEL PROGRAMA

**Aplicación desplegada:**
https://examen-computacion-nube-oscarespino.streamlit.app/

**Repositorio GitHub:**
https://github.com/OsEspino-Sen/examen-computacion-nube-cifar10

## 🧠 Modelo de Inteligencia Artificial

* Dataset: CIFAR-10
* Número de clases: 10
* Entrada: imágenes RGB de 32 × 32 píxeles
* Framework: TensorFlow / Keras
* Arquitectura: Red Neuronal Convolucional (CNN)
* Entrenamiento: Google Colab con GPU T4
* Accuracy obtenida: **62.18%**
* Archivo del modelo: `models/cifar10_model.keras`

## 📷 Funciones del programa

La aplicación permite:

* Subir imágenes JPG, JPEG o PNG.
* Utilizar la cámara del navegador.
* Clasificar la imagen mediante la CNN.
* Mostrar la clase predicha.
* Mostrar el porcentaje de confianza.
* Mostrar las tres predicciones más probables.
* Configurar un umbral de confianza.

## 🏷️ Clases de CIFAR-10

1. Avión
2. Automóvil
3. Pájaro
4. Gato
5. Ciervo
6. Perro
7. Rana
8. Caballo
9. Barco
10. Camión

## ☁️ Flujo del proyecto

```text
Google Colab
      ↓
Entrenamiento de CNN con CIFAR-10
      ↓
cifar10_model.keras
      ↓
Aplicación Streamlit
      ↓
GitHub
      ↓
Streamlit Community Cloud
      ↓
URL pública del programa
```

## 📁 Estructura

```text
examen-computacion-nube-cifar10/
│
├── .streamlit/
│   └── config.toml
│
├── models/
│   └── cifar10_model.keras
│
├── app.py
├── requirements.txt
├── README.md
├── entrenamiento_cifar10_colab.ipynb
└── .gitignore
```

## 💻 Ejecución local

Se utilizó Python 3.11.

```powershell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

La aplicación puede abrirse normalmente en:

```text
http://localhost:8501
```

## 📚 Entrenamiento

El archivo `entrenamiento_cifar10_colab.ipynb` contiene el proceso de entrenamiento realizado en Google Colab:

1. Carga del dataset CIFAR-10.
2. Preparación de los datos.
3. Construcción de la CNN.
4. Entrenamiento del modelo.
5. Evaluación.
6. Guardado del modelo en formato `.keras`.

## 🎓 Información académica

**Institución:** UTH
**Asignatura:** Computación en la Nube
**Proyecto:** Clasificación de imágenes mediante CNN y Streamlit
