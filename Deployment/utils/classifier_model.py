# Attribution: Original code by RUCKBReasoning
# Repository: https://github.com/RUCKBReasoning/codes

"""
Module: schema_item_classifier
Defines a PyTorch model to classify relevant tables and columns for Text-to-SQL tasks.
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, RobertaModel


class SchemaItemClassifier(nn.Module):
    """
    Model for predicting table and column relevance given encoded schema and question.

    Architecture:
    - RoBERTa encoder (pretrained or randomly initialized)
    - BiLSTM + pooling for table and column token sequences
    - Cross-attention between table and column embeddings
    - MLP heads for binary classification of tables and columns
    """
    def __init__(self, model_name_or_path: str, mode: str):
        """
        Initialize encoder and classification heads.

        Args:
            model_name_or_path: HuggingFace model identifier or local path.
            mode: One of ['train', 'eval', 'test'] to control encoder initialization.
        """
        super().__init__()
        # Load or initialize PLM encoder
        if mode in ("eval", "test"):
            config = AutoConfig.from_pretrained(model_name_or_path)
            self.encoder = RobertaModel(config)
        elif mode == "train":
            self.encoder = RobertaModel.from_pretrained(model_name_or_path)
        else:
            raise ValueError(f"Invalid mode '{mode}', choose from 'train', 'eval', 'test'.")

        hidden_size = self.encoder.config.hidden_size
        self.hidden_size = hidden_size

        # Table classification head
        self.table_bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            bidirectional=True
        )
        self.table_mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

        # Column classification head
        self.column_bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            bidirectional=True
        )
        self.column_mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8
        )

    def _bilstm_pool(self, embeddings: torch.Tensor, lstm: nn.LSTM) -> torch.Tensor:
        """
        Run embeddings through BiLSTM and return concatenated final hidden states.

        Args:
            embeddings: Tensor of shape (seq_len, hidden_size)
            lstm: Configured nn.LSTM module.

        Returns:
            Tensor of shape (1, hidden_size)
        """
        outputs, (h_n, _) = lstm(embeddings.unsqueeze(1))  # (seq_len, 1, hidden*2)
        # Take final forward and backward hidden states
        hidden = torch.cat((h_n[-2], h_n[-1]), dim=-1)  # (hidden*2)
        return hidden.unsqueeze(0)  # (1, hidden*2)

    def _apply_cross_attention(
        self,
        table_embs: torch.Tensor,
        col_embs: torch.Tensor,
        col_counts: list
    ) -> torch.Tensor:
        """
        Apply cross-attention from each table to its columns and update table embeddings.

        Args:
            table_embs: (num_tables, hidden_size)
            col_embs: (num_columns, hidden_size)
            col_counts: List with number of columns per table.

        Returns:
            Updated table_embs with same shape.
        """
        updated = []
        start = 0
        for count in col_counts:
            cols = col_embs[start:start+count].unsqueeze(1)  # (count, 1, hidden)
            table = table_embs[len(updated)].unsqueeze(1)     # (1, 1, hidden)
            attn_out, _ = self.cross_attention(table, cols, cols)
            updated.append(attn_out.squeeze(1))
            start += count
        updated = torch.cat(updated, dim=0)
        # Residual and normalize
        return nn.functional.normalize(table_embs + updated, p=2, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        col_token_ids: list,
        tbl_token_ids: list,
        col_counts: list
    ) -> dict:
        """
        Forward pass: encode sequence, pool table/column embeddings, apply cross-attn, classify.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            col_token_ids: list of lists of token indices per column.
            tbl_token_ids: list of lists of token indices per table.
            col_counts: list of number of columns per table.

        Returns:
            Dict with 'table_logits' and 'column_logits' lists.
        """
        batch_size = input_ids.size(0)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_embs = outputs.last_hidden_state  # (batch, seq_len, hidden)

        all_tbl_logits = []
        all_col_logits = []

        for b in range(batch_size):
            emb = seq_embs[b]  # (seq_len, hidden)
            # Pool table embeddings
            tbl_embs = torch.cat([
                self._bilstm_pool(emb[ids], self.table_bilstm)
                for ids in tbl_token_ids[b]
            ], dim=0)
            # Pool column embeddings
            col_embs = torch.cat([
                self._bilstm_pool(emb[ids], self.column_bilstm)
                for ids in col_token_ids[b]
            ], dim=0)

            # Cross-attention update
            tbl_embs = self._apply_cross_attention(tbl_embs, col_embs, col_counts[b])

            # Classification heads
            tbl_logits = self.table_mlp(tbl_embs)
            col_logits = self.column_mlp(col_embs)

            all_tbl_logits.append(tbl_logits)
            all_col_logits.append(col_logits)

        return {
            "batch_table_name_cls_logits": all_tbl_logits,
            "batch_column_info_cls_logits": all_col_logits
        }