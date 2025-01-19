from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from abc import ABC, abstractmethod
from typing import List, Tuple
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer, util
import re

# Descarga de recursos necesarios
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
import numpy as np

class TextPreprocessor:

    @classmethod
    def preprocess(cls, text: str, lang='spanish') -> str:
        text = text.lower()
        # Tokenizar usando NLTK
        tokens = word_tokenize(text)
        # Eliminar stopwords usando NLTK
        stop_words = set(stopwords.words(lang))
        tokens = [t for t in tokens if t not in stop_words]

        return ' '.join(tokens)

class Retriever(ABC):

    def __init__(self, name='abstract_retriever'):
        self.name = name

    def get_name(self):
        return self.name

    """
    Este método recibe un conjunto de documentos y los indexa para poder realizar búsquedas posteriores
    """
    @abstractmethod
    def build_index(self, documents: List[str], lang: str = 'spanish'):
        pass

    """
        Este método búsca los documentos relevantes para una query.
        Devuelve una lista con el la posición (index) del documento encontrado y su score de relevancia.
    """
    @abstractmethod
    def search(self, query: str, top_k: int = 3, lang:str = 'spanish') -> List[Tuple[int, float]]:
        pass

    """
        Este método búsca los documentos relevantes para una query.
        Devuelve los documentos que considera relevantes.
    """
    @abstractmethod
    def search_documents(self, query: str, top_k: int = 3, lang:str = 'spanish') -> List[str]:
        pass


'''
    * Búsqueda eficiente: El uso de NearestNeighbors con una métrica de similitud como el coseno permite realizar búsquedas rápidas.
    * TF-IDF como base: Las palabras más relevantes en cada documento obtienen un peso mayor, mejorando la precisión de la búsqueda.
'''
class SparseRetrieverNM(Retriever):

    def __init__(self):
        super().__init__("sparse_retriever_nm")
        self.vectorizer = TfidfVectorizer()
        self.nn_model = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="auto")

    """
    Construye el índice usando TF-IDF
    """
    def build_index(self, documents: List[str], lang: str = 'spanish'):
        self.documents = documents
        # Limpiar tokens innecesarios
        processed_docs = [TextPreprocessor.preprocess(doc, lang) for doc in self.documents]
        # Generar embeddings dispersos TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
        # Construir un modelo de búsqueda eficiente
        self.nn_model.fit(self.tfidf_matrix)

    def search(self, query: str, top_k: int = 5, lang: str = 'spanish') -> List[Tuple[int, float]]:
        # Vectorizar la consulta
        processed_query = TextPreprocessor.preprocess(query, lang)
        query_vector = self.vectorizer.transform([processed_query])

        # Encontrar los vecinos más cercanos
        distances, indices = self.nn_model.kneighbors(query_vector, n_neighbors=top_k)
        # Retornar resultados como documentos y distancias inversas (para similitud)
        return [(idx, score) for idx, score in zip(indices[0], distances[0])][::-1]

    def search_documents(self, query: str, top_k: int = 3, lang: str = 'spanish') -> List[str]:
        relevant_documents = self.search(query, top_k, lang)
        return [self.documents[idx] for idx, score in relevant_documents]

class SparseRetriever(Retriever):
    """
    Implementa búsqueda dispersa usando BM25.
    """
    def __init__(self):
        super().__init__('sparse_retriever')


    def build_index(self, documents: List[str], lang:str = 'spanish'):
         # Guardar documentos originales
         self.documents = documents
         #  Procesar texto eliminando tokens no relevantes
         processed_docs = [TextPreprocessor.preprocess(doc, lang) for doc in self.documents]
         # Tokenizar para BM25
         tokenized_docs = [doc.split() for doc in processed_docs]
         # Inicializar BM25
         self.bm25 = BM25Okapi(tokenized_docs)


    def search(self, query: str, top_k: int = 3, lang:str = 'spanish') -> List[Tuple[int, float]]:
        """Realiza búsqueda BM25."""
        # Preprocesar query
        processed_query = TextPreprocessor.preprocess(query, lang)
        query_tokens = processed_query.split()
        # Obtener scores BM25
        scores = self.bm25.get_scores(query_tokens)
        # Obtener top_k resultados
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_indices]

    def search_documents(self, query: str, top_k: int = 3, lang:str = 'spanish') -> List[str]:
        relevant_documents = self.search(query, top_k, lang)
        return [self.documents[idx] for idx, score in relevant_documents]

class DenseRetriever(Retriever):

    def __init__(self, model='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__('dense_retriever' + model)
        # Cargar modelo de embeddings multilingüe
        self.model = SentenceTransformer(model)


    def build_index(self, documents: List[str], lang: str = 'spanish'):
        # Guardar documentos originales
        self.documents = documents
        #  Procesar texto eliminando tokens no relevantes
        processed_docs = [TextPreprocessor.preprocess(doc, lang) for doc in self.documents]
        # Generar y almacenar embeddings
        self.embeddings = self.model.encode(processed_docs, show_progress_bar=True)


    def search(self, query: str, top_k: int = 3, lang: str = 'spanish') -> List[Tuple[int, float]]:
        # Realiza búsqueda por similitud de embeddings.
        processed_query = TextPreprocessor.preprocess(query, lang)
        # Generar embedding de la query
        query_embedding = self.model.encode([processed_query])
        # Calcular similitud
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        # Obtener top_k resultados
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(idx, similarities[idx]) for idx in top_indices]


    def search_documents(self, query: str, top_k: int = 3, lang: str = 'spanish') -> List[str]:
        relevant_documents = self.search(query, top_k, lang)
        return [self.documents[idx] for idx, score in relevant_documents]

class HybridRetriever(Retriever):

    def __init__(self, weight_sparse: float = 0.3,
                 weight_dense: float = 0.7, model='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__('hybrid_retriever' + model)
        self.model = model
        self.weight_sparse = weight_sparse
        self.weight_dense = weight_dense

    def build_index(self, documents: List[str], lang: str = 'spanish'):
        self.sparse_retriever = SparseRetriever()
        self.dense_retriever = DenseRetriever(self.model)
        self.sparse_retriever.build_index(documents)
        self.dense_retriever.build_index(documents)
        self.documents = documents

    def search(self, query: str, top_k: int = 3, lang: str = 'spanish') -> List[Tuple[int, float]]:
        # Obtener resultados de ambos retrievers
        sparse_results = self.sparse_retriever.search(query, top_k=top_k, lang=lang)
        dense_results = self.dense_retriever.search(query, top_k=top_k, lang=lang)

        # Combinar scores
        combined_scores = {}
        for idx, score in sparse_results:
            combined_scores[idx] = score * self.weight_sparse

        for idx, score in dense_results:
            if idx in combined_scores:
                combined_scores[idx] += score * self.weight_dense
            else:
                combined_scores[idx] = score * self.weight_dense

        # Ordenar resultados finales
        sorted_results = sorted(combined_scores.items(),
                              key=lambda x: x[1],
                              reverse=True)[:top_k]
        # Preparar resultados
        return [(idx, score) for idx, score in sorted_results]

    def search_documents(self, query: str, top_k: int = 3, lang: str = 'spanish') -> List[str]:
        relevant_documents = self.search(query, top_k, lang)
        return [self.documents[idx] for idx, score in relevant_documents]


class RetrieversFactory:

    @classmethod
    def get_retrievers(cls):
        return [SparseRetriever(), SparseRetrieverNM(), DenseRetriever(), DenseRetriever( model='distiluse-base-multilingual-cased-v1'), HybridRetriever(), HybridRetriever(model='distiluse-base-multilingual-cased-v1')]


import torch
from transformers import pipeline
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pytorch_utils")


class LLMModel:

    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir="./models/TinyLlama-1.1B-Chat-v1.0"):
        self.device_setup = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype = torch.float32 if  self.device_setup == "mps" else (torch.float16 if torch.cuda.is_available() else torch.float32),
            cache_dir=cache_dir,
            local_files_only=False
        ).to( self.device_setup)

    def __answer_with_model(self, prompt, do_sample, temperature, top_p, max_length, show_prompt):
        formatted_prompt = self.tokenizer.apply_chat_template(conversation=prompt, tokenize=False, return_dict=False,  add_generation_prompt=True)

        # Tokenizar
        inputs = self.tokenizer(formatted_prompt, truncation=True, max_length=max_length,return_tensors="pt")
        inputs = {k: v.to( self.device_setup) for k, v in inputs.items()}

         # Muestra infomacion de log
        if show_prompt:
          print(formatted_prompt)
          print("--- Token size: ---")
          [print("\t", k, ": ", len(v[0])) for k, v in inputs.items()]
          print("-------------------")

        # Generar respuesta
        outputs = self.model.generate(
            **inputs,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decodificar y limpiar respuesta
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Limpiar la respuesta para que no aparezca el prompt
        return response.split("<|assistant|>")[1].strip()

    def __get_nocontext_prompt(self, query):
        return [
            {
                "role": "system",
                "content": "Give a comprehensive answer to the question. Respond only to the question asked, response should be concise and relevant to the question. If you don't know the answer respond with 'I do not know the answer'.",
            },
            {"role": "user", "content": "Answer the following question: " + query + "\n"},
        ]

    def __get_context_prompt(self, query, context=""):
        context = "" if not context else context
        return [{
            "role": "system",
            "content": "Use the following context to answer the question concisely and accurately. If the answer cannot be deduced from the context, do not answer the question and just say 'I do not know'. If you can answer with yes or no prioritize short answers.",
        },
        {"role": "user", "content": "Considering the following information:\n"+'\n'.join(context)+"\n answer to this question: "+query+"\n"}
    ]

    def answer(self, query, context, use_context=False, do_sample=True, temperature=0.1,
               top_p=0.9, max_length=2047, show_prompt=False):
        prompt = self.__get_context_prompt(query, context) if use_context else self.__get_nocontext_prompt(query)
        return self.__answer_with_model(prompt, do_sample, temperature, top_p, max_length, show_prompt)


def format_dataset(dataset):
    # Aplanar los documentos si son listas de listas
    documents = []
    for doc in dataset["documents"]:
        if isinstance(doc, list):
            documents.extend(doc)  # Añadir documentos individuales
        else:
            documents.append(doc)
    queries = {dataset["id"][idx]: question for idx, question in enumerate(dataset['question'])}
    relevant_docs = {dataset["id"][idx]: relevat_docs for idx, relevat_docs in enumerate(dataset["documents"])}
    response = {dataset["id"][idx]: response for idx, response in enumerate(dataset["response"])}
    return {'documents': documents, 'queries': queries, 'gold_std': relevant_docs, 'responses': response}


import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import matplotlib.pyplot as plt
import numpy as np
import warnings
import nltk
nltk.download('wordnet')

def evaluate_qa_system(models, dataset):
    results_by_model = {}

    # Métricas y helpers
    vectorizer = TfidfVectorizer()
    rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    for model_name, model, context_usage in models:
        metrics = {
            "bleu": [],
            "rouge": {"rouge1": [], "rouge2": [], "rougeL": []},
            "meteor": [],
            "exact_match": [],
            "f1": [],
            "cosine_similarity": []
        }

        for entry in dataset:
            query = entry["question"]
            context = entry["documents"]
            ground_truth = entry["answer"]

            # Respuesta del modelo
            prediction = model.answer(query=query, context=context, use_context=context_usage)

            # BLEU
            smoothing_function = SmoothingFunction()
            bleu_score = sentence_bleu([ground_truth.split()], prediction.split(), smoothing_function=smoothing_function.method1)
            metrics["bleu"].append(bleu_score)

            # ROUGE
            rouge_scores = rouge_scorer_obj.score(ground_truth, prediction)
            for rouge_metric in ["rouge1", "rouge2", "rougeL"]:
                metrics["rouge"][rouge_metric].append(rouge_scores[rouge_metric].fmeasure)

            # METEOR
            meteor = meteor_score([ground_truth.split()], prediction.split())
            metrics["meteor"].append(meteor)

            # Exact Match
            exact_match = int(prediction.strip() == ground_truth.strip())
            metrics["exact_match"].append(exact_match)

            # F1 Score
            y_true = set(ground_truth.split())
            y_pred = set(prediction.split())
            common = y_true & y_pred
            precision = len(common) / len(y_pred) if y_pred else 0
            recall = len(common) / len(y_true) if y_true else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            metrics["f1"].append(f1)

            # Cosine Similarity
            tfidf_matrix = vectorizer.fit_transform([ground_truth, prediction])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            metrics["cosine_similarity"].append(cosine_sim)

        # Promediar resultados
        avg_metrics = {
            "bleu": np.mean(metrics["bleu"]),
            "rouge": {k: np.mean(v) for k, v in metrics["rouge"].items()},
            "meteor": np.mean(metrics["meteor"]),
            "exact_match": np.mean(metrics["exact_match"]),
            "f1": np.mean(metrics["f1"]),
            "cosine_similarity": np.mean(metrics["cosine_similarity"])
        }
        results_by_model[model_name] = avg_metrics

    return results_by_model

def plot_evaluation_results(results_by_model):
    metrics = ["bleu", "meteor", "exact_match", "f1", "cosine_similarity"]
    models = list(results_by_model.keys())

    for metric in metrics:
        values = [results_by_model[model][metric] for model in models]
        plt.figure()
        plt.bar(models, values)
        plt.title(f"Comparación de {metric.capitalize()}")
        plt.xlabel("Modelos")
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
