import java.util.*;

/*

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
*/
public class MultiHeadAttention {
        private final int dOut;
        private final int numHeads;
        private final int headDim;
        private final Matrix WQuery;
        private final Matrix WKey;
        private final Matrix WValue;
        private final Matrix outProj;
        private final Matrix mask;
        private final double dropoutRate;
        public MultiHeadAttention(int dIn, int dOut, int contextLength, double dropout, int numHeads) {

            if (dOut % numHeads == 0) {
                throw new IllegalArgumentException("num_heads must divide d_out");
            }
            this.dOut = dOut;
            this.numHeads = numHeads;
            this.headDim = dOut / numHeads;
            this.dropoutRate = dropout;

            // Initialize weights
            this.WQuery = Matrix.random(dIn, dOut, 0, 0.02);
            this.WKey = Matrix.random(dIn, dOut, 0, 0.02);
            this.WValue = Matrix.random(dIn, dOut, 0, 0.02);
            this.outProj = Matrix.random(dOut, dOut, 0, 0.02);
            this.mask = new Matrix(contextLength, contextLength);
            for (int i = 0; i < contextLength; i++) {
                for (int j = i + 1; j < contextLength; j++) {
                    this.mask.getData()[i][j] = 1; // 1 means masked
                }
            }
        }

         public Matrix forward (Matrix input){
             int b = input.getRows();
             int numTokens = input.getCols();

             Matrix keys = input.multiply(WKey);
             Matrix queries = input.multiply(WQuery);
             Matrix values = input.multiply(WValue);

             // Reshape into multi-head dimensions: (b, numTokens, numHeads, headDim)
             keys = reshapeForMultiHead(keys, b, numTokens, numHeads, headDim);
             queries = reshapeForMultiHead(queries, b, numTokens, numHeads, headDim);
             values = reshapeForMultiHead(values, b, numTokens, numHeads, headDim);
             
             Matrix keysTransposed = Matrix.transpose(keys);
             Matrix queriesTransposed = Matrix.transpose(queries);
             Matrix valuesTransposed = Matrix.transpose(values);

//             Compute scaled dot-product attention (aka self-attention) with a causal mask
             Matrix attnScores = queries.multiply(keysTransposed);
             attnScores = attnScores.divide(Math.sqrt(headDim));
             attnScores = Matrix.applyMask(attnScores,mask);

             Matrix attnWeights = Matrix.softmax(attnScores);
             attnWeights = Matrix.dropout(attnWeights, dropoutRate);

             // Compute context vectors
             Matrix contextVec = attnWeights.multiply(values);

             // Combine heads back into (b, numTokens, dOut)
             contextVec = combineHeads(contextVec, b, numTokens);

             // Final linear projection
             contextVec = contextVec.multiply(outProj);

             return contextVec;
         }

    private Matrix reshapeForMultiHead(Matrix m, int b, int numTokens, int numHeads, int headDim) {
        // Reshape matrix (b, numTokens * numHeads * headDim) for multi-head attention
        double[][] data = m.getData();
        double[][] reshapedData = new double[b * numHeads][numTokens * headDim / numHeads];

        for (int i = 0; i < b; i++) {
            for (int j = 0; j < numTokens; j++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < headDim; d++) {
                        reshapedData[i * numHeads + h][j * headDim + d] = data[i][j * headDim * numHeads + h * headDim + d];
                    }
                }
            }
        }

        return new Matrix(reshapedData);
    }

    private Matrix combineHeads(Matrix contextVec, int b, int numTokens) {
        // Logic for combining heads (b, numHeads, numTokens, headDim) to (b, numTokens, dOut)
        double[][] data = contextVec.getData();
        double[][] combined = new double[b][numTokens * dOut];

        for (int i = 0; i < b; i++) {
            for (int j = 0; j < numTokens; j++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < headDim; d++) {
                        combined[i][j * dOut + h * headDim + d] = data[i * numHeads + h][j * headDim + d];
                    }
                }
            }
        }

        return new Matrix(combined);
    }
}


/*
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec
 */