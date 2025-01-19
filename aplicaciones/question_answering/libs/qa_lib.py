from libs.rag_lib import TextPreprocessor, SparseRetrieverNM, SparseRetriever, DenseRetriever, HybridRetriever, RetrieversFactory, LLMModel, format_dataset, evaluate_qa_system, plot_evaluation_results
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

def evaluate_qa_system_adv(models, dataset):
    results_by_model = {}

    # Métricas y helpers
    vectorizer = TfidfVectorizer()
    rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    for model_name, model in models:
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
            ground_truth = entry["answer"]

            # Respuesta del modelo
            prediction = model.answer(query=query, do_sample=False)

            # BLEU
            bleu_score = sentence_bleu([ground_truth.split()], prediction.split())
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


def evaluate_qa_answers(answers, gold_std):
    results_by_model = {}

    # Métricas y helpers
    vectorizer = TfidfVectorizer()
    rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    for model_name, predictions in answers.items():
        metrics = {
            "bleu": [],
            "rouge": {"rouge1": [], "rouge2": [], "rougeL": []},
            "meteor": [],
            "exact_match": [],
            "f1": [],
            "cosine_similarity": []
        }

        for idx, prediction in predictions.items():
            ground_truth = gold_std[idx]

            # BLEU
            bleu_score = sentence_bleu([ground_truth.split()], prediction.split())
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


class QuestionAnswering:

    def __init__(self, generator, rag):
        self.generator = generator
        self.rag = rag

    def answer(self, query, lang="spanish", do_sample=True, show_prompt=False,top_k=5, temperature=0.1):
        relevant_documents = self.rag.search_documents(query, lang=lang, top_k=top_k)
        answer = self.generator.answer(query, relevant_documents, show_prompt=show_prompt, use_context=True, do_sample=do_sample, temperature=temperature)
        return answer

class QuestionAnsweringFactory:

    @classmethod
    def get_models(cls, generator, documents):
        retrievers = RetrieversFactory.get_retrievers()
        models = []
        for retriever in retrievers:
            retriever.build_index(documents)
            models.append(("Q&A-" + retriever.get_name(), QuestionAnswering(generator, retriever)))

        return models


