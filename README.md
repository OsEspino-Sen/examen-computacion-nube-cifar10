# Examen - Computación en la Nube | UTH

## Solución propuesta

La arquitectura está diseñada para cumplir exactamente el flujo del examen:

**Google Colab → entrenamiento CNN → archivo `.keras` → GitHub → Streamlit Community Cloud**

Se utiliza **CIFAR-10**, que contiene 50,000 imágenes de entrenamiento y 10,000 de prueba en 10 clases: avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco y camión.

### Estructura

```text
examen_computacion_nube_cifar10/
├── app.py
├── entrenamiento_cifar10_colab.py
├── entrenamiento_cifar10_colab.ipynb
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── models/
    └── cifar10_model.keras   <-- se genera en Colab
```

## Paso 1. Entrenar en Google Colab

1. Abre `entrenamiento_cifar10_colab.ipynb` en Google Colab.
2. Activa GPU si está disponible: **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU**.
3. Ejecuta las celdas en orden.
4. Espera a que termine el entrenamiento.
5. El notebook crea `models/cifar10_model.keras` y descarga ese archivo.

El entrenamiento usa una CNN con aumento de datos, Batch Normalization, Dropout, Early Stopping y reducción automática del learning rate.

## Paso 2. Colocar el modelo

Copia el archivo descargado desde Colab dentro de:

```text
models/cifar10_model.keras
```

No cambies el nombre.

## Paso 3. Probar localmente

Recomendado: Python 3.12.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Instala:

```bash
pip install -r requirements.txt
```

Ejecuta:

```bash
streamlit run app.py
```

La aplicación tendrá dos entradas:

- Subir imagen
- Usar cámara

Y mostrará:

- clase predicha
- porcentaje de confianza
- Top-3 de predicciones
- advertencia cuando la confianza sea baja

## Paso 4. Subir a GitHub

El repositorio debe contener `app.py`, `requirements.txt`, `.streamlit/config.toml` y el modelo dentro de `models/`.

Comandos:

```bash
git init
git add .
git commit -m "Proyecto final computacion en la nube"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
git push -u origin main
```

## Paso 5. Desplegar en Streamlit Community Cloud

1. Entra a `https://share.streamlit.io/`.
2. Conecta GitHub.
3. Selecciona **Create app**.
4. Selecciona tu repositorio.
5. Branch: `main`.
6. Main file path: `app.py`.
7. Selecciona un subdominio fácil de recordar.
8. Presiona **Deploy**.

Si el proyecto está correctamente organizado, Community Cloud instalará las dependencias desde `requirements.txt` y ejecutará `app.py`.

## Importante sobre CIFAR-10

CIFAR-10 no contiene una clase "persona" ni "libros". Sus clases reales son las diez descritas arriba. Por tanto, para que el proyecto sea técnicamente correcto y defendible en la exposición, la interfaz utiliza las clases reales de CIFAR-10.

Una foto moderna de una persona, un libro o un objeto muy distinto del dataset no debe presentarse como una clase válida. El modelo siempre escogerá una de sus 10 categorías, pero una confianza baja indica que la imagen puede no pertenecer bien a la distribución aprendida.

## Cómo explicar el proyecto en la exposición

**1. Dataset:** CIFAR-10.

**2. Entrenamiento:** Google Colab y TensorFlow/Keras.

**3. Modelo:** CNN con capas convolucionales, pooling, Batch Normalization y Dropout.

**4. Guardado:** `cifar10_model.keras`.

**5. Aplicación:** Streamlit recibe la imagen, la convierte a RGB, la redimensiona a 32×32 y la pasa a la CNN.

**6. Resultado:** se muestran clase, confianza y Top-3.

**7. Nube:** el repositorio se conecta a Streamlit Community Cloud y se obtiene una URL pública.

## Problemas típicos

### Error: "El modelo todavía no está en el proyecto"

Falta:

```text
models/cifar10_model.keras
```

### Error al instalar TensorFlow

Confirma que el despliegue use una versión de Python compatible. El `requirements.txt` de este proyecto está preparado para TensorFlow 2.21.0 y Python 3.12.

### La cámara no funciona

El navegador debe tener permiso para utilizar la cámara. También se puede cambiar a "Subir imagen".

### Predicciones extrañas

Es normal si la imagen real tiene objetos diferentes a CIFAR-10. El modelo fue entrenado exclusivamente para sus diez clases.

## Entregables del examen

- URL pública de Streamlit
- Código fuente / repositorio GitHub
- Notebook de Google Colab
- Breve documentación
- Nombre del estudiante visible en la interfaz

Para el último punto, cambia el texto de la interfaz en `app.py` o agrega tu nombre en el pie de página antes de publicar.
