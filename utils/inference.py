import torch
from transformers import BertTokenizer, XLMRobertaTokenizer, MT5Tokenizer
import argparse
from model import MultiEncoderClassifier

def inference(model, texts, device, bert_tokenizer, xlmr_tokenizer, mt5_tokenizer):
    model.eval()
    with torch.no_grad():
        bert_enc = bert_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        xlmr_enc = xlmr_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        mt5_enc  = mt5_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        
        input_ids_bert = bert_enc['input_ids'].to(device)
        attention_mask_bert = bert_enc['attention_mask'].to(device)
        
        input_ids_xlmr = xlmr_enc['input_ids'].to(device)
        attention_mask_xlmr = xlmr_enc['attention_mask'].to(device)
        
        input_ids_mt5 = mt5_enc['input_ids'].to(device)
        attention_mask_mt5 = mt5_enc['attention_mask'].to(device)
        
        # We only use the first output (logits_a) for Task A.
        logits_a, _ = model(input_ids_bert, attention_mask_bert,
                            input_ids_xlmr, attention_mask_xlmr,
                            input_ids_mt5, attention_mask_mt5)
        preds_a = torch.argmax(logits_a, dim=1)
    return preds_a.cpu().numpy()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizers
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    mt5_tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
    
    # Initialize the multi-encoder model (we still load the full model, but ignore Task B output)
    model = MultiEncoderClassifier(num_labels_a=2, num_labels_b=2)
    model.load_state_dict(torch.load(args.input_model, map_location=device))
    model.to(device)
    
    texts = args.texts
    preds_a = inference(model, texts, device, bert_tokenizer, xlmr_tokenizer, mt5_tokenizer)
    label_map_a_inv = {0: "NOT", 1: "OFF"}
    for text, pa in zip(texts, preds_a):
        print(f"Text: {text}\nTask A Prediction: {label_map_a_inv[pa]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, default="multiencoder_model.pt", help="Path to the trained model checkpoint")
    # Pass inference texts as command-line arguments separated by spaces.
    parser.add_argument("--texts", nargs="+", required=True, help="Texts for inference")
    args = parser.parse_args()
    
    main(args)