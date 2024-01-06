import torch.nn as nn 
import torch 

class SelfAttention(nn.Module): 
    def __init__(self, embed_size, heads): 
        super(SelfAttention, self).__init__() 
        self.embed_size = embed_size 
        self.heads = heads 
        self.head_dim = embed_size // heads 

        assert (self.head_dim * heads == embed_size) # must be multiple 

        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False) 
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False) 
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False) 

        self.fc_out = nn.Linear(heads*self.head_dim, embed_size) 
    def forward(self, values, keys, query, mask): 
        N = query.shape[0]  # batch size? 
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1] 
        # Previously, values was of shape (N, value_len, embed_size), where value_len corresponds to length of target sentence?
        values = values.reshape(N, value_len, self.heads, self.head_dim) 
        keys = keys.reshape(N, key_len, self.heads, self.head_dim) 
        query = query.reshape(N, query_len, self.heads, self.head_dim) 

        values = self.values(values) 
        keys = self.keys(keys) 
        query = self.queries(query) 

        scores = torch.einsum("nqhd,nkhd->nhqk", [query, keys])

        # Q shape is (N, query_len, n_heads, head_dim)
        # K shape is (N, key_len, n_heads, head_dim) 
        # QK^T is shape (N, heads, query_len, key_len) 
        # for each query, how good is each respective key 

        if mask is not None: 
            scores = scores.masked_fill(mask == 0, float("-1e25"))
        attention = torch.softmax(scores / (self.embed_size ** (1/2)), dim = 3) 

        # calculate attention*V and then concatenate results from each of the heads 
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim) # in self-attention, key_len == value_len 
        # scores is (N, heads, query_len, key_len) 
        # values is (N, value_len, heads, head_dim) 
        # output is (N, query_len, heads, head_dim) 

        # attention nonlinearity 
        out = self.fc_out(out) 
        return out 
    

class TransformerBlock(nn.Module): 
    def __init__(self, embed_size, heads, dropout, forward_expansion): 
        super(TransformerBlock, self).__init__() 

        # x -> Multi-head att. -> add & norm -> feed forward -> add & norm 
        self.attention = SelfAttention(embed_size, heads) 
        # layer norm does normalization per example, batch norm does normalization per batch 
        self.norm1 = nn.LayerNorm(embed_size) 
        self.norm2 = nn.LayerNorm(embed_size) 

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size), 
            nn.ReLU(), 
            nn.Linear(forward_expansion*embed_size, embed_size) 
        )
        self.dropout = nn.Dropout(dropout) 

    def forward(self, value, key, query, mask): 
        attention = self.attention(value, key, query, mask) 
        # include the skip connection 

        x = self.dropout(self.norm1(attention + query)) 
        forward = self.feed_forward(x) 
        out = self.dropout(self.norm2(forward + x))
        return out 
    
class Encoder(nn.Module): 
    def __init__(self, src_vocab_size, embed_size, n_layers, heads, device, forward_expansion, dropout, max_length): 
        super(Encoder, self).__init__() 
        self.embed_size = embed_size

        # max length is required for positional encodings 

        self.device = device 
        self.embedding = nn.Embedding(src_vocab_size, embed_size) 
        self.positional_encoding = nn.Embedding(max_length, embed_size) 
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, heads, dropout, forward_expansion
                )
            for _ in range(n_layers)]
        ) 
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x, mask): 
        N, seq_length = x.shape 
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.embedding(x) + self.positional_encoding(positions)) 

        for layer in self.layers: 
            out = layer(out, out, out, mask) # self attention 
        return out 


class DecoderBlock(nn.Module): 
    def __init__(self, embed_size, heads, forward_expansion, dropout, device): 
        super(DecoderBlock, self).__init__() 
        self.attention = SelfAttention(embed_size, heads) 
        self.norm = nn.LayerNorm(embed_size) 
        self.transformer_block = TransformerBlock(
                    embed_size, heads, dropout, forward_expansion
                )
        self.dropout = nn.Dropout() 
    
    # src mask -- to prevent computation with padded inputs 
    def forward(self, x, value, key, src_mask, target_mask): 
        # Per the figure in the attention paper, the first decoder block is multi headed attention with the input as the Q, K, and V
        # followed by an add and norm 
        attention = self.attention(x, x, x, target_mask) # target_mask is the one that enforces causality for decoding 
        query = self.dropout(self.norm(attention + x))

        # The previous block generates the queries for the next block (this time, not masked attention )

        # This is defined by  x -> Multi-head att. -> add & norm -> feed forward -> add & norm which is just a transformer block 
        out = self.transformer_block(value, key, query, src_mask) 

        return out 
    
class Decoder(nn.Module): 
    def __init__(self, target_vocab_size, embed_size, n_layers, heads, forward_expansion, dropout, device, max_length): 
        super(Decoder, self).__init__() 
        self.device = device 
        self.embedding = nn.Embedding(target_vocab_size, embed_size) 
        self.positional_encoding = nn.Embedding(max_length, embed_size) 

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(n_layers)]
        )
        self.fc_out = nn.Linear(embed_size, target_vocab_size) 
        self.dropout = nn.Dropout(dropout) 
    
    def forward(self, x, enc_out, src_mask, target_mask): 
        N, seq_length = x.shape 
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.embedding(x) + self.positional_encoding(positions)) 
        for layer in self.layers: 
            # incoming values & keys from encoder are the same 
            out = layer(out, enc_out, enc_out, src_mask, target_mask) 
        out = self.fc_out(out) # convert embed_size to target_vocab_size prediction 
        return out 


class Transformer(nn.Module): # finally 
    def __init__(self, src_vocab_size, target_vocab_size, src_pad_index, target_pad_index, embed_size = 256, n_layers = 6, forward_expansion = 4, heads = 8, dropout = 0, device = "cuda", max_len = 100): 
        super(Transformer, self).__init__() 
        self.encoder = Encoder(
            src_vocab_size, 
            embed_size, 
            n_layers, 
            heads, 
            device, 
            forward_expansion, 
            dropout, 
            max_len 
        )

        self.decoder = Decoder( 
            target_vocab_size, 
            embed_size, 
            n_layers,
            heads, 
            forward_expansion, 
            dropout, 
            device, 
            max_len
        )
        self.src_pad_index = src_pad_index 
        self.target_pad_index = target_pad_index
        self.device = device  

    def make_src_mask(self, src):
        # shape is (N, 1, 1, src_len) 
        src_mask = (src != self.src_pad_index).unsqueeze(1).unsqueeze(2) 
        return src_mask.to(self.device) 
    
    def make_target_mask(self, target): 
        N, target_len = target.shape 
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand( 
            N, 1, target_len, target_len 
        )
        return target_mask.to(self.device) 
    
    def forward(self, source, target): 
        src_mask = self.make_src_mask(source) 
        target_mask = self.make_target_mask(target) 
        enc_out = self.encoder(source, src_mask) 
        out = self.decoder(target, enc_out, src_mask, target_mask) 
        return out 
    


if __name__ == "__main__": 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on " + str(device)) 
    input = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 5, 3, 2, 7, 8, 9, 4]]).to(device) 

    target = torch.tensor([[1, 2, 3, 4, 7, 8, 9, 0], [1, 4, 5, 6, 7, 8, 9, 0]]).to(device) 
    source_pad_index = 0 
    target_pad_index = 0 
    source_vocab_size = 10 
    target_vocab_size = 10 
    model = Transformer(source_vocab_size, target_vocab_size, source_pad_index, target_pad_index).to(device) 
    print(target[:, :-1]) # target should not have the end of sentence token 
    output = model(input, target[:, :-1])
    print(output.shape) 








