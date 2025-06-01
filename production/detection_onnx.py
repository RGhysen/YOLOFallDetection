import cv2
import numpy as np
import onnxruntime as ort

class BaseDetectionModule:
    """
    ONNX-based object detection wrapper around YOLO-NAS-S.
    Initializes a single ORT session and exposes `predict` for numpy images.
    """
    def __init__(
        self,
        model_path: str = 'yolonas_fall_base.onnx',
        intra_op_threads: int = None,
        inter_op_threads: int = 1,
        execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL,
        opt_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        use_xnnpack: bool = False,
        xnnpack_threads: int = None
    ):
        # Class names for mapping labels
        self.class_names = ['fall', 'stand']

        # Configure session options
        opts = ort.SessionOptions()
        if intra_op_threads is not None:
            opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = inter_op_threads
        opts.execution_mode = execution_mode
        opts.graph_optimization_level = opt_level
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True

        # Select providers
        providers = []
        if use_xnnpack:
            providers.append((
                'XNNPACKExecutionProvider',
                {'num_threads': xnnpack_threads}
            ))
        providers.append(('CPUExecutionProvider', {}))

        # Create the session
        self.session = ort.InferenceSession(model_path, opts, providers=providers)
        # Cache input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def frame_predict(
        self,
        frame_rgb: np.ndarray,
        conf_threshold: float = 0.0 # Default to no filtering
    ) -> np.ndarray:
        """
        Run detection on a single BGR image.

        Args:
            frame_bgr: HxWx3 numpy uint8 or float32 image in BGR.
            conf_threshold: confidence threshold to filter boxes.

        Returns:
            detections: Nx6 array of [x1, y1, x2, y2, label, confidence].
        """
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr = frame_bgr.astype(np.uint8)
        frame_bgr = frame_bgr.transpose(2, 0, 1)[None, ...]

        h, w = frame_rgb.shape[:2]

        # Run inference
        output = self.session.run(self.output_names, {self.input_name: frame_bgr})

        detections = output[0]        # [batch_index, x1, y1, x2, y2, confidence, class_index]

        # Early exit if nothing detected
        if detections.size == 0:
            return np.empty((0, 6), dtype=float)

        # Parse columns (ignore detections[:,0] which is batch_index)
        boxes  = detections[:, 1:5]          # x1, y1, x2, y2
        scores = detections[:, 5]            # confidence
        labels = detections[:, 6].astype(int)# class indices

        # Apply confidence threshold
        keep = scores >= conf_threshold
        if not np.any(keep):
            return np.empty((0, 6), dtype=float)

        boxes  = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # Assemble into [x1, y1, x2, y2, label, score]
        out = np.concatenate([
            boxes,
            labels.reshape(-1, 1),
            scores.reshape(-1, 1)
        ], axis=1)

        return out