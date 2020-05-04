def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = triu(ones(attn_shape), k=1).astype("uint8")
    return from_numpy(subsequent_mask) == 0