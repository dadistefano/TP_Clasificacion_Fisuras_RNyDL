# TP Clasificacion de Fisuras en Pavimento

Trabajo Practico Integrador de la materia **Redes Neuronales y Deep Learning**.

Alumno: Diego Alberto Di Stefano  
Legajo: 0134844  
Profesor: Alan Dreszman  
Archivo principal: `Tp_Clasificacion_fisuras_Diego_Di_Stefano.ipynb`

# Link google drive (publico solo lectura): https://drive.google.com/drive/folders/1qXqemA3yZCkBu6mApa1CrslfQBKLljug?usp=sharing 

## Descripcion

El notebook implementa un sistema de clasificacion automatica de fisuras en pavimento a partir de imagenes del dataset **Road Damage Dataset (CC0)**.

El trabajo incluye:

- Carga del dataset desde Google Drive.
- Lectura de anotaciones en formato YOLO.
- Recorte de regiones con fisuras a partir de bounding boxes.
- Division del dataset en entrenamiento, validacion y prueba.
- Preprocesamiento y data augmentation con Albumentations.
- Entrenamiento de un modelo con **Transfer Learning usando MobileNetV2**.
- Fine tuning del modelo MobileNetV2.
- Evaluacion con accuracy, reporte de clasificacion y matriz de confusion.
- Prediccion sobre una imagen individual subida en Colab.
- Entrenamiento adicional de una **CNN personalizada desde cero**.
- Comparacion entre MobileNetV2 inicial, MobileNetV2 con fine tuning y CNN desde cero.

## Clases consideradas

El proyecto clasifica tres tipos de fisuras:

- `Longitudinal Crack`
- `Transverse Crack`
- `Alligator Crack`

La estructura queda preparada para una posible extension futura con una clase adicional `No Crack`.

## Requisitos

El trabajo esta preparado para ejecutarse en **Google Colab**.

Se recomienda usar:

- Python 3
- Runtime con GPU activada
- Google Drive montado

El notebook instala las librerias necesarias en la primera seccion:

```python
!pip install tensorflow keras opencv-python numpy pandas matplotlib scikit-learn albumentations tqdm pyyaml --quiet
```

## Preparacion del dataset

Antes de ejecutar el notebook, el profesor debe tener el archivo comprimido del dataset en Google Drive con esta ruta:

```text
/content/drive/MyDrive/TP_Clasificacion_Fisuras/DataSet/road-damage-dataset.zip
```

Si el archivo se encuentra en otra ubicacion, modificar en el notebook la variable:

```python
dataset_path_drive = '/content/drive/MyDrive/TP_Clasificacion_Fisuras/DataSet/road-damage-dataset.zip'
```

El notebook descomprime el dataset en:

```text
/content/road_damage_dataset/
```

Luego genera los recortes procesados en:

```text
/content/processed_cracks/
```

Tambien copia artefactos importantes a Google Drive dentro de:

```text
/content/drive/MyDrive/TP_Clasificacion_Fisuras/
```

## Como ejecutar

1. Abrir Google Colab.
2. Subir o abrir el archivo `Tp_Clasificacion_fisuras_Diego_Di_Stefano.ipynb`.
3. Ir a `Entorno de ejecucion` > `Cambiar tipo de entorno de ejecucion`.
4. Seleccionar acelerador por hardware: `GPU`.
5. Ejecutar las celdas en orden desde el inicio.
6. Cuando Colab lo solicite, autorizar el montaje de Google Drive.
7. Verificar que el ZIP del dataset exista en la ruta indicada.
8. Continuar ejecutando todas las secciones hasta la evaluacion final y comparacion.

## Archivos generados

Durante la ejecucion se generan archivos locales en Colab y copias en Google Drive.

Modelos:

- `best_model.h5`: mejor modelo MobileNetV2 del entrenamiento inicial.
- `best_fine_tuned_model.h5`: mejor modelo MobileNetV2 luego del fine tuning.
- `best_custom_cnn_model.h5`: mejor modelo de la CNN personalizada desde cero.

Historiales:

- `history_initial_training.pkl`
- `history_combined_fine_tuning.pkl`
- `history_custom_cnn.pkl`

Resultados de comparacion:

- `comparison_curves_val_only.png`
- `comparison_test_metrics.png`
- Archivo CSV con metricas comparativas.

Carpetas principales en Google Drive:

```text
TP_Clasificacion_Fisuras/
|-- DataSet/
|-- processed_cracks/
|-- models/
|-- history_logs/
`-- comparison_results/
```

## Notas importantes

- El notebook debe ejecutarse de arriba hacia abajo para que las variables, DataFrames y generadores queden definidos.
- La seccion de prediccion individual requiere subir manualmente una imagen cuando Colab lo solicita.
- Si se reinicia el entorno de Colab, se deben volver a ejecutar las celdas previas para reconstruir variables como `df_train`, `df_val`, `df_test`, `train_generator` y `test_generator`.
- La CNN desde cero puede obtener metricas inferiores a MobileNetV2 porque no utiliza pesos preentrenados de ImageNet y debe aprender las caracteristicas visuales desde el dataset disponible.
- El modelo principal del trabajo es MobileNetV2 con Transfer Learning y Fine Tuning.

## Resultado esperado

Al finalizar la ejecucion, el notebook muestra:

- Distribucion de clases.
- Ejemplos de imagenes y bounding boxes.
- Curvas de accuracy y loss.
- Accuracy y loss sobre el conjunto de prueba.
- Reporte de clasificacion por clase.
- Matriz de confusion.
- Prediccion de una imagen individual.
- Comparacion grafica y tabular entre los modelos entrenados.
