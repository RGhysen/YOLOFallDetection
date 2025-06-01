from detection_onnx import BaseDetectionModule
from tracker_module import TrackerModule
from fusion_model_onnx import TemporalClassificationModule
import numpy as np
import glob
import os
import argparse
import cv2
import time

frame_count = 0
total_time = 0.0

def run_detection(frame, visualize=False):
    global frame_count, total_time
    """
    Shared helper to: resize, forward, visualize.
    raw_frame should be a HxWx3 numpy array (BGR).
    """
    start_time = time.time()
    frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_LINEAR)

    # Run detection
    dets = BaseDetection.frame_predict(frame) # [N, 6] array of [x1, y1, x2, y2, label, confidence]

    # Run Tracker
    tracks = Tracker.track(dets, frame, conf_threshold=0.3276128537002468)

    tids  = [tid  for tid, feat, bbox in tracks]
    feats = [feat for tid, feat, bbox in tracks]
    bboxes= [bbox for tid, feat, bbox in tracks]

    # Run classification on all tracks
    logits_per_track = {}
    if tids:
        feat_arr   = np.stack(feats, axis=0)
        probs_arr  = Classification.forward_batch(feat_arr, tids)
        for i, tid in enumerate(tids):
            logits_per_track[tid] = probs_arr[i]

    # Compute & draw FPS
    elapsed = time.time() - start_time
    total_time += elapsed
    frame_count += 1

    # Visualize detections and tracks
    if visualize:
        for tid, probs in logits_per_track.items():
            label = int(probs[0] < probs[1])  # 0=fall, 1=stand
            conf = max(probs)
            color = (0, 0, 255) if label == 0 else (0, 150, 0)  # red for fall, green for stand

            # draw bounding box and label
            _, _, bbox = next(x for x in tracks if x[0]==tid)
            x1,y1,x2,y2 = bbox.astype(int)
            tw = x2 - x1
            th = y2 - y1
            
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            txt = f"{('ID'+str(tid)+' ') if tid is not None else ''}{conf:.2f}"
            (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1,y1-th-4), (x1+tw,y1), color, -1)
            cv2.putText(frame, txt, (x1, y1-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255,255,255), 1, cv2.LINE_AA)

        # instantaneous FPS
        fps_inst = 1.0 / elapsed
        cv2.putText(frame, f"FPS: {fps_inst:.2f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
            
        cv2.imshow("feed", frame)
        
def main():
    global frame_count, total_time

    parser = argparse.ArgumentParser(description="Real-time Fall Detection from Video")
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: webcam index (0,1,...) or path to video file"
    )
    parser.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="Enable drawing boxes/text and window display"
    )
    parser.set_defaults(visualize=False)
    args = parser.parse_args()
    
    # Prepare display window for visualization
    cv2.namedWindow("feed", cv2.WINDOW_NORMAL)

    # Open video source
    use_testset = (args.source.lower() == "test")

    # When using video source
    if not use_testset:
        try:
            src = int(args.source)
        except ValueError:
            src = args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source {args.source}")

    # When using test set
    if use_testset:
        TEST_FOLDER = "path_to_test_images" # Test images should be frames of videos. When multiple videos are present, they should be named like "video1_frame001.jpg", "video1_frame002.jpg", etc.
        
        # collect all image files
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(TEST_FOLDER, e)))
        files.sort()

        current_prefix = None
        for img_path in files:
            # determine this frame's "video" prefix
            filename = os.path.basename(img_path)
            prefix = filename.split("_", 1)[0]

            # when prefix changes, reset the detector
            if prefix != current_prefix:
                current_prefix = prefix
                Classification.reset()
                Tracker.reset()
                print(f"Processing video prefix: {current_prefix}")

            # read and process
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            run_detection(frame, visualize=args.visualize)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Running over video source or live feed
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break

            run_detection(frame, visualize=args.visualize)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if frame_count > 0:
        avg_fps = frame_count / total_time
        print(f"Processed {frame_count} frames in {total_time:.2f}s --> Average FPS: {avg_fps:.2f}")

    if not use_testset:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize modules
    BaseDetection = BaseDetectionModule()
    Tracker = TrackerModule()
    Classification = TemporalClassificationModule()
    Classification.reset()

    main()