import cv2
import torch
import torch.nn.functional as F
import numpy as np

def visualize_detections(image_np, bboxes, base_labels, base_conf, speeds, logits, track_ids=None,
                         window_name="feed", save_path=None, fps=None):
    if bboxes.shape[0] == 0 or logits.size == 0:
        return

    vis = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    t_logits = torch.from_numpy(logits).float()
    if t_logits.ndim == 1:
        t_logits = t_logits.unsqueeze(0)
    probs = F.softmax(t_logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    for (x1,y1,x2,y2), pred, conf, speed, tid in zip(
            bboxes.astype(int),
            preds,
            confs,
            speeds,
            (track_ids or [])
        ):
        
        if tid is None:
            continue

        label = "fall" if pred == 0 else "stand"
        color = (0,0,255) if pred == 0 else (0,150,0)

        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)

        txt = f"{('ID'+str(tid)+' ') if tid is not None else ''}{conf:.2f}"
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1,y1-th-4), (x1+tw,y1), color, -1)
        cv2.putText(vis, txt, (x1, y1-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 1, cv2.LINE_AA)
        if fps is not None:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(vis, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 2, cv2.LINE_AA)

    if save_path:
        cv2.imwrite(save_path, vis)
    else:
        cv2.imshow(window_name, vis)
        cv2.waitKey(1)
