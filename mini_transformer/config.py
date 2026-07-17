from dataclasses import dataclass

@dataclass
class Config:
    d_model: int = 32
    n_heads: int = 4
    n_layers: int= 2
    d_ff: int = 128
    max_seq_len: int = 16
    vocab_size: int = 49 
