from tokenizers import Tokenizer
import pickle

from utils import get_tokenizer

# Load your trained tokenizer file
tokenizer = get_tokenizer("./")

# Access the tiktoken encoding object which has the vocab info
# mergeable_ranks is a dict[bytes, int] mapping token bytes to token IDs
mergeable_ranks = tokenizer.enc._mergeable_ranks

# Convert bytes to strings and create stoi/itos mappings
stoi = {}
itos = {}

for token_bytes, token_id in mergeable_ranks.items():
    # Decode bytes to string, replacing invalid UTF-8 sequences
    token_str = token_bytes.decode('utf-8', errors='replace')
    stoi[token_str] = token_id
    itos[token_id] = token_str

# Also include special tokens
for special_token, token_id in tokenizer.enc._special_tokens.items():
    stoi[special_token] = token_id
    itos[token_id] = special_token

meta = {
    "stoi": stoi,
    "itos": itos
}

# Save as meta.pkl
with open("meta.pkl", "wb") as f:
    pickle.dump(meta, f)