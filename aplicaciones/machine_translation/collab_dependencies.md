Instalar en primer lugar ciertas dependencias
'''
!pip install datasets, evaluate, rouge-score
'''
* Cambiar la funcion translate a:

'''
import torch

def translate(text, model, tokenizer, do_sample=False, temperature=1.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"  
    model.to(device) 
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)  
    model.eval()  

    with torch.no_grad():
      translated_tokens = model.generate(**inputs, do_sample=do_sample, temperature=temperature)
      translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
      return translated_text
''' 