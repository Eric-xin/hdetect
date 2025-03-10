# HDetect: Hybrid Hateful language detection

This project implements an offensive language detection system using a multi-encoder architecture. The system leverages three powerful pretrained transformers:

- **BERT (multilingual cased)**
- **XLM-RoBERTa (base)**
- **mT5 (small variant, encoder only)**

The model is designed for **Task A** (offensive language identification) on the OLID dataset. The architecture fuses representations from each encoder and feeds them into a custom classification head.

## Model Architecture

The overall architecture is as follows:

```
                           Input Text
                                │
                ┌───────────────┴───────────────┐
                │         Tokenization        │
                └───────────────┬───────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         ▼                      ▼                      ▼
   [BERT Encoder]        [XLM-RoBERTa Encoder]      [mT5 Encoder]
         │                      │                      │
         ▼                      ▼                      ▼
  Extract [CLS]         Extract first token      Extract first token 
    representation         representation         representation
         │                      │                      │
         └───────────────┬─────────────────────────────┘
                         │ Concatenation│
                         └──────┬───────┘
                                │
                                ▼
                   [Classification Head for Task A]
                                │
                                ▼
                          Output: {NOT, OFF}
```

### Detailed Explanation

1. **Tokenization:**  
   The input text is tokenized by three different tokenizers corresponding to the three encoders. Each tokenizer converts text into input IDs and attention masks.

2. **Encoders:**  
   - **BERT Encoder:** Processes input tokens and outputs contextualized embeddings. We use the [CLS] token as the representation.
   - **XLM-RoBERTa Encoder:** Similarly, it processes the text and extracts the first token’s representation.
   - **mT5 Encoder:** The mT5 model’s encoder is used to produce embeddings from the input text, and we take the first token’s representation.

3. **Fusion:**  
   The representations from the three encoders are concatenated into a single combined vector.

4. **Classification Head:**  
   The concatenated vector is passed through a feed-forward neural network (with dropout and ReLU activation) to produce logits for Task A, which classifies the text as either "NOT" offensive or "OFF" offensive.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:

   ```
   torch
   transformers
   pandas
   tqdm
   tensorboard
   ```

## Usage

### Training

Run the training script to train the model on the OLID dataset for Task A.

```bash
python train.py --train_file path/to/olid-training-v1.tsv --epochs 3 --batch_size 16 --lr 2e-5 --max_length 128 --log_dir logs --output_model multiencoder_model.pt
```

TensorBoard logs will be saved in the specified log directory.

### Testing

Evaluate the trained model on a test dataset:

```bash
python test.py --test_file path/to/olid-test.tsv --batch_size 16 --input_model multiencoder_model.pt
```

### Interactive Inference CLI

Run the inference script for an interactive command-line interface:

```bash
python inference.py --input_model multiencoder_model.pt --texts "I love sunny days!" "You are a horrible person!"
```

You can also run the interactive CLI by executing the inference script without text arguments, then typing input interactively.

## Additional Features

- **TensorBoard Integration:**  
  Training logs (loss per epoch) are saved in a specified log directory for monitoring with TensorBoard.

- **Device Selection:**  
  The code automatically selects CUDA if available (or falls back to CPU/MPS).

- **Zipping Logs:**  
  You can easily add a script to zip your log directory after training if needed.

## Model Architecture Illustration

The model takes an input text and processes it through three separate pretrained encoders (BERT, XLM-RoBERTa, mT5). Their outputs are concatenated to form a comprehensive representation, which is then used by a classification head to determine if the text is offensive (OFF) or not (NOT). This multi-encoder design allows the model to leverage complementary information from different pretrained models, resulting in a more robust representation for offensive language detection.

## Contact

I am happy to help with any questions or concerns. You can reach me at [ericxin123@hotmail.com](mailto:ericxin123@hotmail.com).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.