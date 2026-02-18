import torch
from torch import nn

class GroupQueryAttentionKVCache(nn.Module):
    def __init__(self, d_model, n_heads, context_length, n_groups):
        super().__init__()
        assert d_model % n_heads == 0, 'Error: d_model is not divisible by n_heads!'
        assert n_heads % n_groups == 0, 'Error: n_heads is not divisible by n_groups!'
        
        self.d_model = d_model
        self.d_h = d_model // n_heads
        self.n_heads = n_heads
        self.context_len = context_length
        self.n_groups = n_groups
        self.group_size = n_heads // n_groups
            
        self.W_query = nn.Linear(self.d_model, self.d_model)
        self.W_key = nn.Linear(self.d_model, self.n_groups * self.d_h)
        self.W_value = nn.Linear(self.d_model, self.n_groups * self.d_h)
        self.W_out = nn.Linear(self.d_model, self.d_model)

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def forward(self, x, use_cache=False):
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        q = self.W_query(x).view(batch_size, seq_len, self.n_heads, self.d_h).transpose(1,2) # (b, n_heads, group_size, seq_len, d_h)
        
        k_new = self.W_key(x).view(batch_size, seq_len, self.n_groups, self.d_h).transpose(1,2) # (b, n_groups, seq_len, d_h)
        v_new = self.W_value(x).view(batch_size, seq_len, self.n_groups, self.d_h).transpose(1,2)

        if use_cache:
            if self.cache_k is not None:
                self.cache_k = torch.concat([self.cache_k, k_new], dim=2)
                self.cache_v = torch.concat([self.cache_v, v_new], dim=2)
            else:
                self.cache_k, self.cache_v = k_new, v_new
            k_bygroup, v_bygroup = self.cache_k, self.cache_v          # (b, n_groups, cached_len, d_h)
            
        else:
            k_bygroup, v_bygroup = k_new, v_new
            self.cache_k, self.cache_v = None, None
            
        k = k_bygroup.repeat_interleave(self.group_size, dim=1) # (b, n_heads, cached_len, d_h)
        v = v_bygroup.repeat_interleave(self.group_size, dim=1) # (b, n_heads, cached_len, d_h)
        
        cached_len = k.shape[2]
        kdotq = q @ k.transpose(-1,-2) / (self.d_h ** 0.5) # (b, n_heads, seq_len, cached_len)

        upper_mask = (torch.arange(cached_len-seq_len-self.context_len,cached_len-self.context_len).unsqueeze(-1) >= torch.arange(cached_len).unsqueeze(0)) 
        lower_mask = (torch.arange(cached_len-seq_len,cached_len).unsqueeze(-1) < torch.arange(cached_len).unsqueeze(0))
        mask = upper_mask + lower_mask
        masked_kdotq = kdotq.masked_fill_(mask, -1e9)  # (b, n_heads, seq_len, cached_len)

        att_scores = torch.softmax(masked_kdotq, dim=-1)
        out = att_scores @ v  # (b, n_heads, seq_len, d_h)
        out = out.transpose(1,2) # (b, seq_len, n_heads, d_h)
        out = out.contiguous().view(batch_size, seq_len, self.d_model) 
        out = self.W_out(out)

        return out
        
    def reset_kvcache(self):
        self.cache_k, self.cache_v = None, None


class FeedForward(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.Linear1 = nn.Linear(d_in, d_hidden)
        self.Linear2 = nn.Linear(d_hidden, d_in)

    def forward(self, x):
        x = self.Linear1(x)
        x = nn.ReLU()(x)
        x = self.Linear2(x)
        return x


class LayerNormalization(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        me, var = x.mean(dim=-1, keepdim=True), x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - me)/(torch.sqrt(var+self.eps))
        return x_norm * self.weight + self.bias


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        var = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(var + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['emb_dim']
        self.d_hidden = config['hidden_dim']
        self.n_heads = config['n_heads']
        self.context_length = config['context_length']
        self.n_kv_groups = config['n_kv_groups']
        
        self.norm1 = RMSNorm(self.d_model)
        self.norm2 = RMSNorm(self.d_model)
        self.attention_layer = GroupQueryAttentionKVCache(self.d_model, self.n_heads, self.context_length, self.n_kv_groups)
        self.ff_layer = FeedForward(self.d_model, self.d_hidden)

    def forward(self, x, use_cache=False):
        x = x + self.attention_layer(self.norm1(x), use_cache) # Pre-layernorm
        x = x + self.ff_layer(self.norm2(x))
        return x

        
class RoPE(nn.Module):
    def __init__(self, d_model, base = 10000.0):
        super().__init__()
        assert d_model % 2 == 0, "Error: d_model must be divisible by 2"
        self.d_model = d_model
        self.theta = torch.tensor([np.exp(-2*(i//2)*np.log(base)/self.d_model) for i in range(self.d_model)])

    def forward(self, x):
        seq_len = x.shape[1]
        x_reshaped = x.view(x.shape[0], x.shape[1], self.d_model//2, 2)
        x_new = (x_reshaped @ torch.tensor([[0,1],[-1,0]], dtype=x.dtype)).view(x.shape[0], x.shape[1], x.shape[2])
        
        cos = torch.cos(torch.arange(1,seq_len+1).unsqueeze(1) * self.theta.unsqueeze(0)) # (seq_len * d_model)
        sin = torch.sin(torch.arange(1,seq_len+1).unsqueeze(1) * self.theta.unsqueeze(0))

        return x * cos + x_new * sin

        