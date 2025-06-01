import onnxruntime as rt
import numpy as np

class TemporalClassificationModule:
    """
    ONNX-accelerated replacement for the PyTorch fusion_modelv2.
    Maintains per-track hidden states, frame-based pruning, and exposes
    forward_cell and reset methods with the same API.
    """
    def __init__(self,
                 model_path= "temporal_head_int8_probs.onnx",
                 pruning_lag: int = 6,
                 num_threads: int = 6):
        # Initialize ONNX Runtime session
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True
        providers = [
            #("XNNPACKExecutionProvider", {"num_threads": num_threads}),
            ("CPUExecutionProvider", {})
        ]
        self.sess = rt.InferenceSession(model_path, opts, providers=providers)

        # Pruning and state
        self.pruning_lag   = pruning_lag   # max frames before clearing hidden state
        self.ages          = {}            # track_id -> frame age count
        self.hidden_states = {}            # track_id -> np.ndarray [1, hidden_dim]

        # Infer hidden dimension from ONNX input
        hidden_input = self.sess.get_inputs()[1]
        self.hidden_dim = hidden_input.shape[1]

    def forward_batch(self, feats: np.ndarray, tids: list[int]):
        """
        Process a batch of features for multiple track IDs in one ONNX call.

        Args:
            feats: np.ndarray of shape [N, feat_dim]
            tids:  list of length N with track IDs

        Returns:
            logits_arr: np.ndarray of shape [N, num_classes]
        """
        # 1) Age all tracks and reset ages for these tids
        for track_id in list(self.ages.keys()):
            self.ages[track_id] += 1
        for tid in tids:
            self.ages[tid] = 0

        # 2) Prune stale tracks
        stale = [tid for tid, age in self.ages.items() if age > self.pruning_lag]
        for tid in stale:
            self.ages.pop(tid, None)
            self.hidden_states.pop(tid, None)

        # 3) Ensure a hidden state exists for each tid
        h_prev_list = []
        for tid in tids:
            if tid not in self.hidden_states:
                self.hidden_states[tid] = np.zeros((1, self.hidden_dim), dtype='float32')
            h_prev_list.append(self.hidden_states[tid])

        # 4) Stack inputs
        feat_arr   = feats.astype('float32')                        # [N, feat_dim]
        h_prev_arr = np.concatenate(h_prev_list, axis=0)           # [N, hidden_dim]

        # 5) Single ONNX call
        probs_arr, h_next_arr = self.sess.run(
            ['probs', 'h_next'],
            {'feat': feat_arr, 'h_prev': h_prev_arr}
        )

        # 6) Scatter back hidden states
        for i, tid in enumerate(tids):
            # keep shape [1, hidden_dim]
            self.hidden_states[tid] = h_next_arr[i:i+1]

        return probs_arr

    def reset(self):
        """
        Clear all track ages and hidden states (e.g. between sequences).
        """
        self.ages.clear()
        self.hidden_states.clear()
