# Curso de Introducción al Procesamiento del Lenguaje Natural (PLN)

Este repositorio contiene una colección de notebooks y recursos para el curso de Procesamiento de Lenguaje Natural. El curso está diseñado para estudiantes y profesionales que desean adentrarse en el mundo del PLN, comenzando desde
conceptos básicos hasta técnicas más avanzadas.

## 🎯 Objetivos del Curso

- Comprender los fundamentos de Procesamiento de Lenguaje Natural
- Aprender a trabajar con diferentes tipos de corpus lingüísticos
- Desarrollar habilidades prácticas en el preprocesamiento de texto
- Implementar modelos básicos de PLN
- Familiarizarse con las principales bibliotecas y herramientas del ecosistema PLN

## 📚 Contenido

### Módulo 1: Preparación de Datos
- [Expresiones Regulares](notebooks/01_expresiones_regulares.ipynb)
- [Preprocesamiento de Datos](notebooks/01_Preprocesamiento_Datos.ipynb)
- [Análisis de Valoraciones](notebooks/01_analisis_de_valoraciones.ipynb)

### Módulo 2: Modelos n-gramas
- [Modelos N-gramas](notebooks/02_modelos_ngramas.ipynb)
- [Naive Bayes](notebooks/02_naive_bayes.ipynb)
- [Regresion Logística](notebooks/02_regresion_logistica.ipynb)

### Módulo 3: Modelos Vectoriales
- [Word2Vec](notebooks/03_word2vec.ipynb)
- [Sherlock Holmes](notebooks/03_embeddings_sherlock_holmes.ipynb)
- [Wikipedia](notebooks/03_exercise_embeddings_wikipedia.ipynb)

### Módulo 4: Modelos Probabilísticos de Tópicos
- [LDA](notebooks/04_LDA_Cordis.ipynb)
- [Extensiones LDA](notebooks/04_Extensiones_LDA.ipynb)

### Módulo 5: Modelos Transformers
- [MLP](notebooks/05_MLP.ipynb)
- [Redes de Neuronas](notebooks/05_Red_Neuronas_Keras.ipynb)
- [Transformers](notebooks/05_Transformers_con_Keras.ipynb)
- [GPT](notebooks/05_Transformers_GPT.ipynb)
- [HuggingFace](notebooks/05_Transformers_HuggingFace.ipynb)

### Módulo 6: Ajuste Fino (fine-tuning)
- [Clasificacion](notebooks/06_Ajuste_Fino_Clasificacion_IMDB.ipynb)
- [Named Entity Recognition (NER)](notebooks/06_Ajuste_Fino_NER.ipynb)

### Módulo 7: Prompting
- [Aprendizaje por Contexto](notebooks/07_Ajuste_por_Instrucciones.ipynb)
- [Evaluación](notebooks/07_Evaluacion_Modelos_Prompts.ipynb)

### Módulo 8: Retrieval Augmented Generation (RAG)
- [Búsqueda Dispersa y Densa](notebooks/08_Busqueda_Dispersa_y_Densa.ipynb)
- [RAG Avanzado](notebooks/08_RAG_Avanzado.ipynb)


## 📊 Datasets Incluidos

- [Valoraciones de Restaurantes](datasets/valoraciones_restaurante.json) - Este dataset contiene reseñas de restaurantes en español, ideal para practicar análisis de texto y expresiones regulares.

## 🛠️ Requisitos Técnicos

- Python 3.7+
- Jupyter Notebook o JupyterLab
- Bibliotecas principales:
    - nltk
    - pandas
    - numpy
    - matplotlib
    - scikit-learn

## 🚀 Comenzando

1. Clona este repositorio:
```bash
git clone https://github.com/cbadenes/curso-pln.git
```
2. Instala las dependencias necesarias:
```bash
pip install -r requirements.txt
```
3. Descarga los recursos necesarios de NLTK:
```bash
import nltk
nltk.download('reuters')
nltk.download('punkt')
nltk.download('punkt_tab')
```
4. Abre los notebooks en Jupyter:
```bash
jupyter notebook
```

## 📖 Estructura de los Notebooks
Cada notebook sigue una estructura similar:

1. Introducción teórica al concepto
2. Implementación práctica
3. Ejercicios y ejemplos
4. Referencias adicionales

## 👥 Contribuciones
Las contribuciones son bienvenidas. Si deseas contribuir:

1. Haz fork del repositorio
2. Crea una nueva rama para tu funcionalidad
3. Envía un pull request

## 📄 Licencia
Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

## 📬 Contacto
Para dudas, sugerencias o colaboraciones, no dudes en:

- Abrir un issue en este repositorio
- Contactar a través de [carlos.badenes](mailto:carlos.badenes@upm.es)

## 🙏 Agradecimientos

- A la comunidad NLTK por sus excelentes recursos y documentación
- A todos los contribuidores y estudiantes que ayudan a mejorar este material

⭐️ Si este curso te resulta útil, no dudes en darle una estrella al repositorio.
