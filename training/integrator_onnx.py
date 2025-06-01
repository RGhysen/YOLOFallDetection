import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
from yolox.tracker.bytetrack_inputIDX import BYTETracker
from detectionv2 import frame_predict
from visualizer import visualize_detections
import time
last_time = time.time()

# ByteTrack configuration.
# (!ByteTrack source code is modified to carry on an inputted tracking index: return track.det_idx!)
class ByteTrackArgument:
    track_thresh = 0.5 # Minimum score for tracking initiation.
    track_buffer = 50 # Number of frames to keep lost tracklets, should correspond to pruning_lag.
    match_thresh = 0.8 # Matching threshold. Application is less crowded video feed so can be set high because lower risk of fake identity association.
    aspect_ratio_thresh = 10
    min_box_area = 100 # 320x320 frames with lowest bounding box ground truth +- 100 pixels area
    mot20 = True # True because Bytetrack is trained on MOT20 dataset

class YoloNASSpeedFusion(nn.Module):
    """
    Integrates:
      1) frame_predict (YOLO-NAS person detection),
      2) BYTETracker multi-object tracking,
      3) Speed/acceleration & bbox-ratio feature extraction,
      4) FusionModule (GRU+MLP) for stand/fall logits,
      5) Optional visualization.
    """
    def __init__(self,
                 fusion_module: nn.Module,
                 tracker_args,
                 conf_threshold: float = 0.5,
                 speed_average_frames: int = 5,
                 visualize: bool = True):
        super().__init__()
        self.visualize = visualize
        self.conf_threshold = conf_threshold
        self.fusion_module = fusion_module
        self.tracker = BYTETracker(tracker_args)

        self.prev_head_positions = defaultdict(lambda: deque(maxlen=speed_average_frames))
        self.speed_average_frames = speed_average_frames

        self.prev_speed = defaultdict(lambda: deque(maxlen=speed_average_frames))

    def forward(self, images: torch.Tensor, targets=None):
        """
        images: Tensor[B, C, H, W] of uint8 [0..255] frames
        Returns:
          bboxes_list:  list of length B, each a [M,4] tensor
          logits_list:  list of length B, each a [M,2] tensor
        """
        B = images.shape[0]
        bboxes_list = []
        logits_list = []
        last_time = time.time()

        for i in range(B):
            # Detect persons in this frame
            image_np = images[i].permute(1,2,0).cpu().numpy().astype(np.uint8)
            dets_np  = frame_predict(
                image_np,
                conf_threshold=self.conf_threshold
            )
            # if no persons, return empty
            if dets_np.shape[0] == 0:
                bboxes_list.append(torch.empty((0,4), device=images.device))
                logits_list.append(torch.empty((0,2), device=images.device))
                continue

            # Track update
            det_indices = np.arange(dets_np.shape[0])
            # ByteTrack only wants [x1,y1,x2,y2,score], so drop our extra label dim
            dets_for_track = dets_np[:, [0,1,2,3,5]]
            tracks = self.tracker.update(
                dets_for_track,
                det_indices,
                image_np.shape[:2],
                image_np.shape[:2]
            )

            person_bboxes = torch.tensor(dets_np[:,:4], device=images.device, dtype=torch.float32)
            bboxes_list.append(person_bboxes)
            M = dets_np.shape[0]

            logits_frame = torch.zeros((M,2), device=images.device)
            speeds_np    = np.zeros((M,), dtype=float)

            # Prune fusion head’s hidden states for stale tracks
            active_tids = {t.track_id for t in tracks if t.track_id is not None}
            self.fusion_module.prune(active_tids)

            # For each track, compute features & run fusion head
            for t in tracks:
                if t.det_idx is None:
                    # track exists but no matching detection -> skip
                    continue
                idx, tid = t.det_idx, t.track_id
                x,y,w,h   = t.tlwh # top-left + width/height

                # Vertical head speed over last N frames
                hist_y = self.prev_head_positions.setdefault(tid, deque(maxlen=self.speed_average_frames))
                hist_y.append(y)
                if len(hist_y) > 1:
                    diffs = [hist_y[j] - hist_y[j-1] for j in range(1, len(hist_y))]
                    raw_speed = sum(diffs) / len(diffs)
                else:
                    raw_speed = 0.0
                norm_speed = raw_speed * 10.0 / (h + 1e-6)

                # Acceleration = Δnorm_speed
                prev_spd = self.prev_speed.get(tid, 0.0)
                acc = (norm_speed - prev_spd) * 10
                self.prev_speed[tid] = norm_speed

                # Bounding-box aspect ratio (height/width)
                bbox_ar = 1.7 * h / (w + 1e-6)

                # Base detection label and confidence
                base_conf = float(dets_np[idx, 5])
                base_label = float(dets_np[idx, 4])

                # Build feature tensor
                feat   = torch.tensor([[norm_speed, acc, bbox_ar, base_label, base_conf]],
                                      dtype=torch.float32,
                                      device=images.device)
                logit  = self.fusion_module.forward_cell(feat, tid, device=images.device)

                logits_frame[idx] = logit
                speeds_np[idx]    = norm_speed

            # collect this frame’s fusion logits
            logits_list.append(logits_frame)

            # visualization
            if self.visualize:
                # align arrays & lists
                b_np      = person_bboxes.cpu().numpy()
                base_label = dets_np[:,4]
                base_conf  = dets_np[:,5]
                scores_np = logits_frame.detach().cpu().numpy()
                track_ids = [None]*M
                for t in tracks:
                    if t.det_idx is not None:
                        track_ids[t.det_idx] = t.track_id
                current_time = time.time()
                fps = 1.0 / (current_time - last_time)
                last_time = current_time

                visualize_detections(
                    image_np,
                    b_np,
                    base_label,
                    base_conf,
                    speeds_np,
                    scores_np,
                    track_ids=track_ids,
                    fps=fps,
                )
                last_time = current_time

        return bboxes_list, logits_list
    
    def reset(self):
        # clear fusion‐module’s GRU bookkeeping for training over multiple video's
        self.fusion_module.reset()
        # reset speed‐history and prev‐speed
        self.prev_head_positions = defaultdict(
            lambda: deque(maxlen=self.speed_average_frames)
        )
        self.prev_speed = {}