from yolox.tracker.bytetrack_inputIDX import BYTETracker # Imports the modified BYTETracker class
from collections import defaultdict, deque
import numpy as np

class ByteTrackArgument:
    track_thresh = 0.5      # Minimum score for tracking initiation.
    track_buffer = 50       # Number of frames to keep lost tracklets, should correspond to pruning_lag.
    match_thresh = 0.7071101107043911      # Matching threshold. Application is less crowded video feed so can be set high because lower risk of fake identity association.
    aspect_ratio_thresh = 10
    min_box_area = 100      # 320x320 frames with lowest bounding box ground truth +- 100 pixels area
    mot20 = True            # True because Bytetrack is trained on MOT20 dataset


class TrackerModule:
    """
    Integrates:
      1) frame_predict (YOLO-NAS person detection),
      2) BYTETracker multi-object tracking,
      3) Speed/acceleration & bbox-ratio feature extraction,
    """
    def __init__(self):
        self.tracker = BYTETracker(ByteTrackArgument())
        self.speed_average_frames = 20
        self.prev_head_positions = defaultdict(lambda: deque(maxlen=self.speed_average_frames))
        self.prev_speed = defaultdict(lambda: deque(maxlen=self.speed_average_frames))

    def reset(self):
        """
        Completely reset the tracker state between videos.
        """
        self.__init__()

    def track(self, dets, image, conf_threshold: float = 0.5) -> np.ndarray:
        """
        Args:
            frame: HxWx3 BGR uint8 image.
            conf_threshold: Confidence threshold for detections.
        Returns:
            Nx10 array [Tracking ID, norm_speed, acc, bbox_ar, base_label, base_conf, x1, y1, x2, y2].
        """

        det_indices = np.arange(dets.shape[0])
        # ByteTrack only wants [x1,y1,x2,y2,score], so drop our extra label dim
        dets_for_track = dets[:, [0,1,2,3,5]]   # -> [x1,y1,x2,y2,conf]
        tracks = self.tracker.update(
            dets_for_track,     # now shape (N,5)
            det_indices,        # detection indices
            image.shape[:2], # frame size (h,w)
            image.shape[:2]
        )

        # --- For each track, compute temporal information carriers ---
        feats = []
        for t in tracks:
            if t.det_idx is None:
                # track exists but no matching detection → skip
                continue
            idx, tid = t.det_idx, t.track_id
            x,y,w,h   = t.tlwh  # top-left + width/height

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
            acc = (norm_speed - prev_spd)
            self.prev_speed[tid] = norm_speed

            # Bounding-box aspect ratio (height/width)
            bbox_ar = 3 * h / (w + 1e-6)

            # Base detection label and confidence
            base_conf = float(dets[idx, 5])
            base_label = float(dets[idx, 4]) 

            # Build feature array
            feat   = np.array([norm_speed, acc, bbox_ar, base_label, base_conf], dtype=np.float32)
            bbox     = dets[t.det_idx, :4].astype(np.float32)
            feats.append((tid, feat, bbox))

        return feats


