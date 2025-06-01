# (speed, acceleration, bbox_aspect_ratio, base_pred, base_pred_conf) → [logit_fall, logit_stand]

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class SpeedClassificationModule(nn.Module):
    def __init__(self,
                 feat_dim: int = 5,
                 hidden_dim: int = 128,
                 num_hidden_layers: int = 0,
                 dropout_p: float = 0.00,
                 pruning_lag: int = 50,
                 ):
        super().__init__()

        # Recurrent core (per-track)
        # GRUCell lets us update one (tid, time) pair at a time
        self.gru_cell = nn.GRUCell(input_size=feat_dim, hidden_size=hidden_dim)

        # Regularization
        self.dropout = nn.Dropout(dropout_p)

        # Deep fusion head
        self.mlp_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_hidden_layers)
        ])

        # Final binary classifier
        self.fc = nn.Linear(hidden_dim, 2)

        # Hidden-state bookkeeping for pruning
        # Maps track_id -> hidden tensor
        self.hidden_states = {}
        # tid -> #frames since last seen
        self.ages          = {}
        # How many frames before we drop an “inactive” track
        self.pruning_lag = pruning_lag

    def forward_cell(self, feat: torch.Tensor, tid: int, device: torch.device = torch.cpu) -> torch.Tensor:
        """
        Update & classify a single detection for track `tid` at the current frame.

        Args:
          feat: Tensor[1, feat_dim] of raw features for this detection.
          tid:  Unique track ID (int).
          device: same device to allocate/init hidden states.
        Returns:
          logits: Tensor[1,2] giving [fall, stand] scores.
        """
        # Fetch or initialize hidden state (no detach here!)
        h_prev = self.hidden_states.get(
            tid,
            torch.zeros(1, self.gru_cell.hidden_size, device= device, dtype=feat.dtype)
        )
        h_prev = h_prev.detach() # limit the gradient graph to a single GRU step rather than back-propagating through all video frames

        # Recurrent update + dropout
        h = self.gru_cell(feat, h_prev)
        h = self.dropout(h)

        # Pass through MLP stack
        for layer in self.mlp_layers:
            h = layer(h)
            h = F.relu(h)
            h = self.dropout(h)

        # Final classification logits
        logits = self.fc(h)  # [1, 2]

        p_fall = logits.softmax(-1)[0,0]

        # Bookkeeping for pruning
        # store the new hidden state & timestamp
        self.hidden_states[tid] = h
        self.ages[tid] = 0

        return logits

    def prune(self, active_tids: set):
        """
        Called once per frame with the set of track_ids seen this frame.
        Increments all ages, resets ages for active ones, and drops
        any tid whose age exceeds pruning_lag.
        """
        # Increase age for all existing tracks
        for tid in list(self.ages.keys()):
            self.ages[tid] += 1

        # Reset age for tracks we just saw
        for tid in active_tids:
            self.ages[tid] = 0

        # Drop stale tracks
        stale = [tid for tid, age in self.ages.items() if age > self.pruning_lag]
        for tid in stale:
            self.ages.pop(tid, None)
            self.hidden_states.pop(tid, None)

    def reset(self):
        """
        Clear ALL tracks—call at the start of each new video so that
        hidden states don’t bleed across videos.
        """
        self.hidden_states.clear()
        self.ages.clear()