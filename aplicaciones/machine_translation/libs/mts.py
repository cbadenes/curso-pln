from abc import ABC, abstractmethod
from transformers import MarianTokenizer, MarianMTModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


device_setup = "cuda" if torch.cuda.is_available() else "cpu"

class AbstractMT(ABC):

    @abstractmethod
    def translate(self, text, do_sample=True, temperature=0.1):
        pass


    def translate_batch(self, texts, do_sample=True, temperature=0.1):
        return [self.translate(text, do_sample=do_sample, temperature=temperature) for text in texts]

class MarianMT(AbstractMT):

    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-es", cache_dir="./models/opus-mt-en-es"):
        global device_setup
        self.model = MarianMTModel.from_pretrained(model_name, cache_dir=cache_dir).to(device_setup)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=cache_dir)


    def translate(self, text, do_sample=True, temperature=0.1):
        global device_setup
        # Tokenizar la entrada
        inputs_tokenized = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device_setup)
        # Traducir
        translated_tokens = self.model.generate(**inputs_tokenized, max_length=200, do_sample=do_sample,  temperature=temperature)
        # Decodificar salida
        return self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)


class T5MT(AbstractMT):

    def __init__(self, model_name="vgaraujov/t5-base-translation-en-es", cache_dir="./models/t5-base-translation-en-es"):
        global device_setup
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=False).to(device_setup)

        self.tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=False)


    def translate(self, text, do_sample=True, temperature=0.1):
        global device_setup
        # Tokenizar la entrada
        inputs_tokenized = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device_setup)
        # Traducir
        translated_tokens = self.model.generate(**inputs_tokenized, max_length=200, do_sample=do_sample, temperature=temperature)
        # Decodificar salida
        return self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

def load_pretrained_marian_mt():
    global device_setup
    marian_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es", cache_dir="./models/pretrained_marian_en-es", local_files_only=False).to(device_setup)
    marian_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es", cache_dir="./models/pretrained_marian_en-es", max_length=512, num_beams= 4,local_files_only=False)
    return marian_model, marian_tokenizer

def load_pretrained_t5():
    global device_setup
    # También existen t5-base o t5-large
    t5_model = T5ForConditionalGeneration.from_pretrained("vgaraujov/t5-base-translation-en-es", cache_dir="./models/pretrained_t5_en-es", local_files_only=False).to(device_setup)
    t5_tokenizer = T5Tokenizer.from_pretrained("vgaraujov/t5-base-translation-en-es", cache_dir="./models/pretrained_t5_en-es", local_files_only=False)
    return t5_model, t5_tokenizer
