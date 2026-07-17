import numpy as np
from mini_transformer.config import Config

# im doint random initialization of embedding tables for tokens and positions, it is not capable for 
# model training, it is just for demonstration purposes.
# In practice, you would want to use a proper initialization method and train the embeddings along with the model.

vocab_size = Config.vocab_size
d_model = Config.d_model
max_seq_len = Config.max_seq_len


token_emb_table = np.random.randn(vocab_size, d_model) * 0.02
pos_emb_table = np.random.randn(max_seq_len, d_model) * 0.02

# fucntion to get embeddings for a sequence of token indices
def get_embeddings(token_indices):
    seq_len = len(token_indices)
    token_embeddings = token_emb_table[token_indices]
    pos_embeddings = pos_emb_table[:seq_len]
    return token_embeddings + pos_embeddings


"""
i = [28, 25, 32, 32, 35] - Hello
embeddings = get_embeddings(i)
print(embeddings)
print(embeddings.shape) # (5, 32)
"""