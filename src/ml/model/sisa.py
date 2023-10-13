import torch
import torch.nn as nn

from torch import Tensor


class AUGRUCell(nn.Module):
    """ https://github.com/waxxyybb/DIN-pytorch/blob/master/rnn.py """

    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True):
        super(AUGRUCell, self).__init__()

        gate_input_dim = input_dim + hidden_dim
        self.reset_gate = nn.Sequential(nn.Linear(gate_input_dim, hidden_dim, bias=bias), nn.Sigmoid())
        self.update_gate = nn.Sequential(nn.Linear(gate_input_dim, hidden_dim, bias=bias), nn.Sigmoid())
        self.h_hat_gate = nn.Sequential(nn.Linear(gate_input_dim, hidden_dim, bias=bias), nn.Tanh())

    def reset_parameters(self):
        std = 1.0 / torch.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x: Tensor, h_prev: Tensor, attention_score: Tensor):
        temp_input = torch.cat([h_prev, x], dim=-1)

        reset = self.reset_gate(temp_input)
        update = self.update_gate(temp_input)

        h_hat = self.h_hat_gate(torch.cat([h_prev * reset, x], dim=-1))
        # print(f'AUGRUCell - update: {update.shape} / attention_score: {attention_score.shape}')
        update = attention_score * update
        h_current = ((1. - update) * h_prev) + (update * h_hat)

        return h_current


class InteractionEncoder(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.rand(emb_dim, emb_dim))
        self.softmax = nn.Softmax(dim=1)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=emb_dim, num_layers=1, bias=True, batch_first=True)
        self.augru = AUGRUCell(input_dim=emb_dim, hidden_dim=emb_dim, bias=True)

    def forward(self, session_emb: Tensor, target_emb: Tensor):
        # print(f'InteractionEncoder - session_emb : {session_emb.shape} / target_emb : {target_emb[:, None, :].shape}')
        target_emb = target_emb[:, None, :].transpose(2, 1)
        s_state, h_state = self.gru(session_emb)

        # Attention
        attention = torch.matmul(s_state, self.w)
        attention = torch.matmul(attention, target_emb)
        attention = self.softmax(attention)

        h_state = h_state.squeeze(0)

        for i in range(s_state.size(1)):  # through sequence
            h_state = self.augru(x=s_state[:, i, :], h_prev=h_state, attention_score=attention[:, i, :])

        return h_state


class SessionEncoder(nn.Module):

    def __init__(self, emb_dim: int):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4, batch_first=False)

    def forward(self, session_emb, padding_mask, src_mask):
        session_emb = self.transformer(
            src=session_emb.transpose(0, 1), src_mask=src_mask, src_key_padding_mask=padding_mask, is_causal=False
        )
        return session_emb.transpose(0, 1)


class SessionInterestSelfAttention(nn.Module):

    def __init__(self, num_item: int, emb_dim: int, max_len: int, pad_id: int):
        super().__init__()
        self.item_embedding = nn.Embedding(num_embeddings=num_item, embedding_dim=emb_dim, padding_idx=pad_id)
        self.positional_embedding = nn.Embedding(num_embeddings=max_len, embedding_dim=emb_dim)

        self.session_encoder = SessionEncoder(emb_dim=emb_dim)
        self.interaction_encoder = InteractionEncoder(emb_dim=emb_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=emb_dim * 2, out_features=emb_dim, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=emb_dim, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, session_items, padding_mask, target_item, src_mask):
        pos_emb = self.positional_embedding(torch.arange(0, session_items.size(1)))
        session_emb = self.item_embedding(session_items) + pos_emb
        target_emb = self.item_embedding(target_item)

        session_emb = self.session_encoder(
            session_emb=session_emb, padding_mask=padding_mask, src_mask=src_mask
        )
        # print(f'session_emb : {session_emb[0, 0, :]} / target_emb : {target_emb[0]}')
        feature = self.interaction_encoder(session_emb=session_emb, target_emb=target_emb)
        # feature = torch.cat([session_emb[:, 0, :], target_emb], dim=-1)
        feature = torch.cat([feature, target_emb], dim=-1)

        return self.feed_forward(feature)
