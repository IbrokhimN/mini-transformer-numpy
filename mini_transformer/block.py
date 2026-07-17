from mini_transformer.layers import res_attention_block, feed_forward_block

def transformer_block(x, W_Q, W_K, W_V, W_O, n_heads, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2):
    #сначала внимание
    x = res_attention_block(x, W_Q, W_K, W_V, W_O, n_heads, gamma1, beta1)
    #потом feed forward
    x = feed_forward_block(x, W1, b1, W2, b2, gamma2, beta2)
    return x