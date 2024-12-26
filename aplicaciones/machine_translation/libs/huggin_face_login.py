from huggingface_hub import login

def hugging_face_log():
    token = ""
    print("Hugging Face logging")
    login(token)