"""
LBKT Model: Language-Based Knowledge Tracing (Li et al., 2024)

Full LBKT model combining:
- BERT Transformer architecture (for semantic understanding)
- Rasch Model Embeddings (for difficulty modeling)
- LSTM Layer (for sequential knowledge tracking)

Key components:
- Rasch Embedding: E_rasch = E_token + E_diff * (E_token + E_segment)
- BERT Encoder: Multi-head attention + Feed-forward
- LSTM: Sequential state tracking for long sequences (>400 interactions)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention' mechanism used by BERT.
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention mechanism as described in "Attention is All You Need".
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0, f"d_model ({d_model}) must be divisible by h ({h})"

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections: d_model => h x d_k
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        # Apply attention
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # Concatenate and project
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class LayerNorm(nn.Module):
    """Layer normalization module."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    Residual connection followed by layer normalization.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit activation function (used by BERT).
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    """Feed-forward network used in Transformer blocks."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class TransformerBlock(nn.Module):
    """
    Single Transformer block with multi-head attention and feed-forward.
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class TokenEmbedding(nn.Embedding):
    """Token embedding layer with padding support."""

    def __init__(self, vocab_size, embed_size=128):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings as described in "Attention is All You Need".
    """

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class SegmentEmbedding(nn.Embedding):
    """Segment embedding for question/skill differentiation."""

    def __init__(self, vocab_size, embed_size=128):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class RaschBERTEmbedding(nn.Module):
    """
    LBKT Embedding with Rasch model difficulty calculation.

    Implements: E_rasch = E_token + E_diff * (E_token + E_segment)

    The Rasch model explicitly models item difficulty:
    - E_token: Question/interaction embedding
    - E_segment: Response/skill embedding  
    - E_diff: Learned difficulty parameter per question
    - E_position: Sinusoidal positional encoding

    Final embedding = Rasch embedding + Positional embedding
    """

    def __init__(self, vocab_size, embed_size, max_seq_len=100, dropout=0.1):
        """
        Args:
            vocab_size: Number of unique questions/skills (n_skill)
            embed_size: Embedding dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        # Token embedding: handles both question IDs and question+correct pairs
        # vocab_size * 2 + 1 to handle: q_id for wrong, q_id + n_skill for correct
        self.token = TokenEmbedding(vocab_size=2 * vocab_size + 1, embed_size=embed_size)
        
        # Positional embedding (max_len - 1 because we use shifted sequences)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=max_seq_len)
        
        # Segment embedding (for question IDs - acts as difficulty in Rasch formulation)
        self.segment = SegmentEmbedding(vocab_size=vocab_size + 1, embed_size=embed_size)
        
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        """
        Args:
            sequence: Token IDs (batch_size, seq_len) - encoded as q_id + correct * n_skill
            segment_label: Question IDs for segment embedding (batch_size, seq_len)

        Returns:
            Embedded sequence with Rasch-style embedding
        """
        # Token and segment embeddings
        token_emb = self.token(sequence)
        segment_emb = self.segment(segment_label)
        
        # Rasch-style embedding from BERT-Rasch notebook:
        # x = token + segment * (token + segment)
        # This uses segment embedding as a multiplicative modifier (like difficulty)
        rasch_emb = token_emb + segment_emb * (token_emb + segment_emb)
        
        # Add positional embedding
        pos_emb = self.position(sequence)
        
        # Final embedding
        x = rasch_emb + pos_emb
        
        return self.dropout(x)


class LBKT(nn.Module):
    """
    LBKT: Language-Based Knowledge Tracing (Li et al., 2024)

    A hybrid model combining:
    1. Rasch Model Embeddings for difficulty modeling
    2. BERT Transformer for semantic understanding
    3. LSTM for sequential knowledge state tracking

    Designed for long-sequence student data (>400 interactions).
    """

    def __init__(
        self,
        num_questions,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        lstm_hidden=128,
        lstm_layers=1,
        max_seq_len=100,
        dropout=0.1,
    ):
        """
        Args:
            num_questions: Number of unique questions/skills
            embed_dim: Hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            num_layers: Number of Transformer blocks
            lstm_hidden: LSTM hidden size
            lstm_layers: Number of LSTM layers
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()

        self.model_name = "lbkt"
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_questions = num_questions
        self.lstm_hidden = lstm_hidden

        # Feed-forward hidden size (4x hidden per BERT paper)
        self.feed_forward_hidden = embed_dim * 4

        # Rasch-based BERT Embedding
        self.embedding = RaschBERTEmbedding(
            vocab_size=num_questions,
            embed_size=embed_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, embed_dim * 4, dropout)
                for _ in range(num_layers)
            ]
        )

        # LSTM for sequential tracking (key for long sequences)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Prediction layer
        self.pred = nn.Linear(lstm_hidden, 1)

    def forward(self, x, segment_info):
        """
        Args:
            x: Input sequence (batch_size, seq_len)
               Encoded as: q_id + correct * n_skill
            segment_info: Question IDs (batch_size, seq_len)
               Used for difficulty embedding and masking

        Returns:
            Predictions (batch_size, seq_len) - sigmoid logits
        """
        # Create attention mask for padding (tokens > 0 are valid)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # Embedding with Rasch model
        x = self.embedding(x, segment_info)

        # Transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        # LSTM for sequential tracking
        x, (h_n, c_n) = self.lstm(x)

        # Prediction
        x = self.pred(x)

        return x.squeeze(-1)


class LBKTSimple(nn.Module):
    """
    Simplified LBKT variant without transformer layers.
    Uses only Rasch embeddings + LSTM for faster training.
    """

    def __init__(
        self,
        num_questions,
        embed_dim,
        lstm_hidden=None,
        lstm_layers=1,
        max_seq_len=512,
        dropout=0.1,
    ):
        """
        Args:
            num_questions: Number of unique questions/concepts
            embed_dim: Embedding dimension
            lstm_hidden: LSTM hidden size (defaults to embed_dim)
            lstm_layers: Number of LSTM layers
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()

        self.model_name = "lbkt_simple"
        self.num_questions = num_questions
        self.embed_dim = embed_dim
        self.lstm_hidden = lstm_hidden or embed_dim

        # Interaction embedding (question + response)
        self.interaction_emb = nn.Embedding(num_questions * 2 + 1, embed_dim, padding_idx=0)
        
        # Question embedding for difficulty
        self.question_emb = nn.Embedding(num_questions + 1, embed_dim, padding_idx=0)
        
        # Difficulty embedding (Rasch component)
        self.difficulty_emb = nn.Embedding(num_questions + 1, embed_dim, padding_idx=0)
        
        # Positional embedding
        self.position_emb = PositionalEmbedding(d_model=embed_dim, max_len=max_seq_len)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(self.lstm_hidden, num_questions)

    def forward(self, q, r):
        """
        Args:
            q: Question IDs (batch_size, seq_len)
            r: Response correctness (batch_size, seq_len) - 0 or 1

        Returns:
            Predictions (batch_size, seq_len, num_questions)
        """
        r = r.long()
        
        # Interaction encoding: q_id for wrong, q_id + num_questions for correct
        x = q + self.num_questions * r
        
        # Embeddings
        interaction_emb = self.interaction_emb(x)
        question_emb = self.question_emb(q)
        diff_emb = self.difficulty_emb(q)
        pos_emb = self.position_emb(x)
        
        # Rasch embedding: E_rasch = E_interaction + E_diff * (E_interaction + E_question)
        rasch_emb = interaction_emb + diff_emb * (interaction_emb + question_emb)
        
        # Add positional encoding
        emb = rasch_emb + pos_emb
        
        # LSTM
        h, _ = self.lstm(emb)
        h = self.dropout(h)
        
        # Output
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y


# Backward compatibility alias
BERT = LBKT
