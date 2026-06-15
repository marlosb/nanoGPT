from tokenizers import Tokenizer
import pickle

# Load your trained tokenizer file
tokenizer = Tokenizer.from_file("data/mytokenizer/")

# The tokenizer vocab:
vocab = tokenizer.get_vocab()

# stoi: string -> int mapping
stoi = vocab

# itos: int -> string mapping (make sure the order matches indices)
# get_vocab() returns the mapping as token->index
itos = {index: token for token, index in vocab.items()}

# Optionally, you may want to explicitly handle special tokens here if needed

meta = {
    "stoi": stoi,
    "itos": itos
}

# Save as meta.pkl
with open("data/mytokenizer/meta.pkl", "wb") as f:
    pickle.dump(meta, f)