import torch
from transformers import BertTokenizer, XLMRobertaTokenizer, MT5Tokenizer
from model import MultiEncoderClassifier  # Adjust import path if needed

# Automatically select device: CUDA > MPS > CPU.
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load the trained model checkpoint.
model = MultiEncoderClassifier(num_labels_a=2)
state_dict = torch.load("model-trained/multiencoder_model.pt", map_location=device)
model.load_state_dict(state_dict, strict=False)
# non strict loading
model.to(device)
model.eval()  # Set model to evaluation mode.

# Initialize tokenizers.
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
mt5_tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')

def inference(model, text, device, bert_tokenizer, xlmr_tokenizer, mt5_tokenizer, max_length=128):
    """Run inference on a single text and return the prediction for Task A."""
    model.eval()
    with torch.no_grad():
        # Wrap text in a list for tokenization.
        bert_enc = bert_tokenizer([text], padding=True, truncation=True, return_tensors='pt', max_length=max_length)
        xlmr_enc = xlmr_tokenizer([text], padding=True, truncation=True, return_tensors='pt', max_length=max_length)
        mt5_enc  = mt5_tokenizer([text], padding=True, truncation=True, return_tensors='pt', max_length=max_length)
        
        # Move tokenized inputs to the device.
        input_ids_bert = bert_enc['input_ids'].to(device)
        attention_mask_bert = bert_enc['attention_mask'].to(device)
        
        input_ids_xlmr = xlmr_enc['input_ids'].to(device)
        attention_mask_xlmr = xlmr_enc['attention_mask'].to(device)
        
        input_ids_mt5 = mt5_enc['input_ids'].to(device)
        attention_mask_mt5 = mt5_enc['attention_mask'].to(device)
        
        # Run inference; only Task A output is used.
        logits_a = model(input_ids_bert, attention_mask_bert,
                            input_ids_xlmr, attention_mask_xlmr,
                            input_ids_mt5, attention_mask_mt5)
        preds_a = torch.argmax(logits_a, dim=1)
    return preds_a.cpu().numpy()[0]

# Define mapping for Task A labels.
label_map_a_inv = {0: "NOT", 1: "OFF"}

# Interactive CLI loop.
print("\nInteractive Inference CLI (Task A). Type 'exit' to quit.")
while True:
    text = input("Enter text for inference: ").strip()
    if text.lower() == "exit" or text == "":
        break
    pred = inference(model, text, device, bert_tokenizer, xlmr_tokenizer, mt5_tokenizer)
    print(f"Task A Prediction: {label_map_a_inv[pred]}\n")