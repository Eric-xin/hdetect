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

"""
NotImplementedError: The operator 'aten::cumsum.out' is not currently implemented
for the MPS device. If you want this op to be added in priority during the 
prototype phase of this feature, please comment on 
https://github.com/pytorch/pytorch/issues/77764.
"""

# Load the trained model checkpoint.
model = MultiEncoderClassifier(num_labels_a=2, num_labels_b=2)
state_dict = torch.load("model-trained/multiencoder_model.pt", map_location=device)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()  # Set model to evaluation mode.

# Initialize tokenizers.
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
mt5_tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')

def inference(model, texts, device, bert_tokenizer, xlmr_tokenizer, mt5_tokenizer, max_length=128):
    model.eval()
    with torch.no_grad():
        # Tokenize texts using each model's tokenizer.
        bert_enc = bert_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
        xlmr_enc = xlmr_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
        mt5_enc  = mt5_tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
        
        # Move tokenized inputs to the selected device.
        input_ids_bert = bert_enc['input_ids'].to(device)
        attention_mask_bert = bert_enc['attention_mask'].to(device)
        
        input_ids_xlmr = xlmr_enc['input_ids'].to(device)
        attention_mask_xlmr = xlmr_enc['attention_mask'].to(device)
        
        input_ids_mt5 = mt5_enc['input_ids'].to(device)
        attention_mask_mt5 = mt5_enc['attention_mask'].to(device)
        
        # Run inference; we only use the Task A output (first output).
        logits_a, _ = model(input_ids_bert, attention_mask_bert,
                            input_ids_xlmr, attention_mask_xlmr,
                            input_ids_mt5, attention_mask_mt5)
        preds_a = torch.argmax(logits_a, dim=1)
    return preds_a.cpu().numpy()

# Define sample texts for inference.
texts = [
    "I love sunny days!",
    "You are a horrible person!"
]

# Run inference and map numeric predictions to labels.
predictions = inference(model, texts, device, bert_tokenizer, xlmr_tokenizer, mt5_tokenizer)
label_map_a_inv = {0: "NOT", 1: "OFF"}

for text, pred in zip(texts, predictions):
    print(f"Text: {text}\nTask A Prediction: {label_map_a_inv[pred]}\n")