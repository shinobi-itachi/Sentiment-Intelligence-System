from transformers import BertTokenizer

def create_bert_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer