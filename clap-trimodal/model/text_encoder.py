from transformers import RobertaModel, AutoModel

def get_text_encoder(name: str, access_token: str):
    if name == "RoBERTA":
        return RobertaModel.from_pretrained("FacebookAI/roberta-base", token=access_token)
    elif name == "DistilRoBERTa":
        return AutoModel.from_pretrained("distilroberta-base", token=access_token)
    else:
        raise ValueError(f"Unsupported text encoder name: {name}")
