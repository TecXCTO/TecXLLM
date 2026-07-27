# TecXLLM
Technology Engineering Computation Expansion  LLM

## 1. Repository Directory Structure
```
my-custom-llm/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # Automated tests for tokenization and model shape
├── data/
│   ├── raw/                   # Raw text files, symbols, math datasets
│   └── tokenized/             # Processed token ID binaries (.bin) for fast loading
├── src/
│   ├── __init__.py
│   ├── tokenizer.py           # <--- PUT THE TOKENIZATION CODE GENERATED ABOVE HERE
│   ├── embeddings.py          # Token embeddings and Rotary Positional Embeddings (RoPE)
│   ├── model.py               # Transformer block, RMSNorm, SwiGLU, and Attention logic
│   ├── dataset.py             # PyTorch Custom Dataset and DataLoader for byte tokens
│   └── train.py               # Main pre-training loops, loss tracking, and checkpoints
├── configs/
│   └── base_config.json       # Hyperparameters (vocab_size, hidden_dim, layers, heads)
├── tests/
│   ├── test_tokenizer.py      # Tests for UTF-8 symbol parsing and token recovery
│   └── test_model.py          # Tests to verify model matrix shapes match up
├── checkpoins/                # Directory to save trained model weight matrices (.pt)
├── .gitignore                 # Excludes heavy datasets and model weights from GitHub
├── LICENSE                    # Open-source license (e.g., MIT or Apache 2.0)
├── README.md                  # Project documentation, setup steps, and architecture map
└── requirements.txt           # Python dependencies (torch, regex, numpy, etc.)

```

2. Breakdown of Key Files and Naming

src/tokenizer.py (Tokenization Code)

Purpose: This is the precise home for the ByteLevelBPETokenizer code generated in the previous step.
Role: It will be imported directly by your data processing pipelines and your main training loop to convert text strings to integer matrices.

src/embeddings.py

Purpose: Manages vector lookups. It takes token integer outputs from tokenizer.py and maps them into deep continuous vectors, while adding Rotary Positional Embeddings (RoPE) so the model understands the spatial ordering of mathematical digits and letters.

src/model.py
Purpose: Houses the actual neural network structure. This contains the Decoder-Only Transformer blocks, Grouped-Query Attention (GQA), Root Mean Square Normalization (RMSNorm), and the SwiGLU activation layers.

src/dataset.py

Purpose: Uses PyTorch’s Dataset module to memory-map text files, break them into blocks matching your context window length (e.g., 1024 tokens), and feed tensors to your hardware.

src/train.py
Purpose: The central execution engine. It initializes your tokenizer, sets up the model architecture, processes the text corpus, computes cross-entropy loss, and optimizes neural weights.

3. Basic Dependency Configuration (requirements.txt)
 
   To ensure this workspace installs cleanly on any machine, populate your requirements.txt with these core libraries:
   
texttorch>=2.0.0
regex>=2023.0.0
numpy>=1.24.0
tqdm>=4.65.0
