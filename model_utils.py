from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

MODEL_PATH = "./ShishirAI_NepWav2Vec2"

model = None
tokenizer = None

def load_model_and_tokenizer():
    global model, tokenizer
    if model is None:
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
    if tokenizer is None:
        tokenizer = Wav2Vec2Tokenizer.from_pretrained(MODEL_PATH)

def get_model():
    return model

def get_tokenizer():
    return tokenizer
