import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # multiply by sqrt of model size as noted in the original paper
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len, d_model) # initialize the size of positional encoder
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # has shape (seq_len, 1)
        denom = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # denominator, with size d_model/2
        pe[:,::2] = torch.sin(pos*denom)
        pe[:,1::2] = torch.cos(pos*denom)
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        
        self.register_buffer('pe',pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1)) # multiply
        self.bias = nn.Parameter(torch.zeros(1)) # add
        self.eps = eps
        
    def forward(self, x):
        me, std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True)
        return self.alpha * (x - me)/(std * self.eps) + self.bias
    
class PositionwiseFF(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(nn.ReLU()(self.linear_1(x))))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model is not divisible by n_head"
        self.d_h = d_model//n_heads # dim of key, query, value
        
        self.LinearQ = nn.Linear(d_model, d_model)
        self.LinearK = nn.Linear(d_model, d_model)
        self.LinearV = nn.Linear(d_model, d_model)
        self.Linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def attention(self, query, key, value, mask, dropout: nn.Dropout):
        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(self.d_h)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)  # (Batch size, n_heads, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return attention_scores @ value, attention_scores   # return shape (Batch size, n_heads, seq_len, d_h) after multiply value
        
        
    def forward(self, q, k, v, mask):
        # q,k,v (batch_size, seq_len, d_model)
        q_proj = self.LinearQ(q)
        k_proj = self.LinearK(k)
        v_proj = self.LinearV(v)
        
        q_proj = q_proj.view(q_proj.shape[0], q_proj.shape[1], self.n_heads, self.d_h).transpose(1, 2)  # (batch size, n_heads, seq_len, d_h)
        k_proj = k_proj.view(k_proj.shape[0], k_proj.shape[1], self.n_heads, self.d_h).transpose(1, 2)
        v_proj = v_proj.view(v_proj.shape[0], v_proj.shape[1], self.n_heads, self.d_h).transpose(1, 2)
        
        x, self.attention_scores = self.attention(q_proj, k_proj, v_proj, mask, self.dropout)
        x = x.transpose(1,2)  # (Batch size, seq_len, n_heads, d_h) 
        x = x.contiguous().view(x.shape[0], x.shape[1], self.d_model)
        return self.Linear(x) # (Batch size, seq_len, d_model)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, selfAttention: MultiHeadAttention, feedForward: PositionwiseFF, dropout: float) -> None:
        super().__init__()
        self.selfAttention = selfAttention
        self.feedForward = feedForward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.selfAttention(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feedForward)
        return x
        
class Encoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, selfAttention: MultiHeadAttention, crossAttention: MultiHeadAttention, feedForward: PositionwiseFF, dropout: float) -> None:
        super().__init__()
        self.selfAttention = selfAttention
        self.crossAttention = crossAttention
        self.feedForward = feedForward
        self.residual_connetions = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connetions[0](x, lambda x: self.selfAttention(x, x, x, tgt_mask))
        x = self.residual_connetions[1](x, lambda x: self.crossAttention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connetions[2](x, self.feedForward)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, 
                 encoder: Encoder, 
                 decoder: Decoder, 
                 src_embed: InputEmbeddings, 
                 tgt_embed: InputEmbeddings, 
                 src_pos: PositionalEncoding, 
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer,
                ) -> None:
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        src = self.encoder(src, src_mask)
        return src
        
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        tgt = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return tgt
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, 
                      tgt_vocab_size: int, 
                      src_seq_len: int, 
                      tgt_seq_len: int,
                      d_model: int = 512,
                      n_blocks: int = 6,
                      n_heads: int = 8,
                      dropout: float = 0.1,
                      d_ff: int = 2048
                     ) -> Transformer:
    # create embedding layer
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # create encoder and decoder
    encoder = Encoder(nn.ModuleList([EncoderBlock(MultiHeadAttention(d_model, n_heads, dropout), PositionwiseFF(d_model, d_ff, dropout), dropout) for _ in range(n_blocks)]))
    decoder = Decoder(nn.ModuleList([DecoderBlock(MultiHeadAttention(d_model, n_heads, dropout), MultiHeadAttention(d_model, n_heads, dropout), PositionwiseFF(d_model, d_ff, dropout), dropout)
                                        for _ in range(n_blocks)
                                    ]))
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    #transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # initialize params
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
    
