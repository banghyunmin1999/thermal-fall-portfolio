import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import threading
import queue
import collections

# nvtx 마커 (없으면 더미 함수)
try:
    from nvtx import push_range, pop_range
except ImportError:
    def push_range(msg): pass
    def pop_range(): pass

RTSP_IR_URL = 'rtsp://admin:Soundmind88!!@192.168.110.231/stream0'
SAMPLE_VIDEO_PATH = "/home/soundmind/fallwatch-thermal/sampleVideo/sample.mp4"
MODEL_PATH = '/home/soundmind/fallwatch-thermal/model/engine/best0625_v3_fp16.engine'

class_names = {0: 'caution', 1: 'normal', 2: 'warning'}
colors = {0: (0, 255, 255), 1: (0, 255, 0), 2: (0, 0, 255)}

def letterbox(img, new_shape=(320, 320), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, dw, dh

class VideoCamera:
    TIME_WINDOW_SECONDS = 10
    DANGER_RATIO_THRESHOLD = 0.7
    ALERT_COOLDOWN_SECONDS = 30

    def __init__(self):
        print("🚀 카메라 및 TensorRT 엔진 초기화 시작...")
        self.cuda_context = pycuda.autoinit.context
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.h_input = None
        self.h_output = None
        self.d_input = None
        self.d_output = None
        self.cap = None
        self.is_running = False
        self.detection_history = collections.deque()
        self.last_alert_time = {'warning': 0, 'caution': 0}
        self._last_displayed_alert = ""
        self.current_counts = {0: 0, 1: 0, 2: 0}
        self.current_alert_message = ""

        try:
            with open(MODEL_PATH, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
                self.context = self.engine.create_execution_context()
                self.input_tensor_name = self.engine.get_tensor_name(0)
                output_tensor_names = [
                    self.engine.get_tensor_name(i)
                    for i in range(self.engine.num_io_tensors)
                    if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
                ]
                if not output_tensor_names:
                    raise ValueError("엔진에서 출력 텐서를 찾을 수 없습니다.")
                self.output_tensor_name = output_tensor_names[0]
                input_shape = self.engine.get_tensor_shape(self.input_tensor_name)
                self.output_shape = self.engine.get_tensor_shape(self.output_tensor_name)
                self.input_height, self.input_width = input_shape[2], input_shape[3]
                self.h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
                self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)
                self.d_input = cuda.mem_alloc(self.h_input.nbytes)
                self.d_output = cuda.mem_alloc(self.h_output.nbytes)
                self.context.set_tensor_address(self.input_tensor_name, int(self.d_input))
                self.context.set_tensor_address(self.output_tensor_name, int(self.d_output))
                print("✅ TensorRT 엔진 로드 및 설정 완료.")
        except Exception as e:
            print(f"❌ TensorRT 엔진 로드 또는 설정 실패: {e}")
            self.release_resources()
            raise

        gst_pipeline = (
            f"rtspsrc location={RTSP_IR_URL} latency=0 protocols=udp ! "
            f"rtph264depay ! h264parse ! nvv4l2decoder ! "
            f"nvvidconv ! video/x-raw, format=BGRx, width=320, height=240 ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false max-buffers=1"
        )

        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print("❌ GStreamer RTSP 카메라 연결 실패. 샘플 비디오로 대체합니다.")
            self.cap = cv2.VideoCapture(SAMPLE_VIDEO_PATH)
        if not self.cap.isOpened():
            self.release_resources()
            raise IOError("카메라 또는 샘플 비디오를 열 수 없습니다.")
        print("✅ 카메라 연결 완료.")
        self.is_running = True
        print("✅ 초기화 완료. 프레임 스트리밍 준비 완료.")

        # 파이프라인 큐와 스레드
        self.capture_q = queue.Queue(maxsize=4)
        self.preprocess_q = queue.Queue(maxsize=4)
        self.infer_q = queue.Queue(maxsize=4)
        self.output_q = queue.Queue(maxsize=2)
        self.latest_result = None
        self.result_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.preprocess_thread = threading.Thread(target=self._preprocess_loop, daemon=True)
        self.infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self.postprocess_thread = threading.Thread(target=self._postprocess_loop, daemon=True)
        self.capture_thread.start()
        self.preprocess_thread.start()
        self.infer_thread.start()
        self.postprocess_thread.start()

    def postprocess(self, frame, predictions, original_w, original_h, r, dw, dh):
        conf_threshold = 0.25
        nms_threshold = 0.45
        boxes = []
        confidences = []
        class_ids = []
        annotated_frame = frame.copy()
        for detection in predictions:
            box = detection[:4]
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                # letterbox 패딩/비율을 반영한 복원
                cx = (box[0] - dw) / r
                cy = (box[1] - dh) / r
                w = box[2] / r
                h = box[3] / r
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                boxes.append([x1, y1, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        highest_priority_class_id = 1  # 기본은 Normal

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cls_id = class_ids[i]
                conf = confidences[i]
                label = f'{class_names.get(cls_id, "Unknown")} {conf:.2f}'
                color = colors.get(cls_id, (255, 255, 255))
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(annotated_frame, label, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if cls_id == 2:
                    highest_priority_class_id = 2
                elif cls_id == 0 and highest_priority_class_id != 2:
                    highest_priority_class_id = 0

        self.update_and_check_alerts(highest_priority_class_id)
        return annotated_frame

    def update_and_check_alerts(self, current_class_id):
        current_time = time.time()
        self.detection_history.append((current_time, current_class_id))
        while self.detection_history and current_time - self.detection_history[0][0] > self.TIME_WINDOW_SECONDS:
            self.detection_history.popleft()
        total_detections = len(self.detection_history)
        if total_detections == 0:
            self.current_counts = {0: 0, 1: 0, 2: 0}
            self.current_alert_message = ""
            self._last_displayed_alert = ""
            return
        counts = {0: 0, 1: 0, 2: 0}
        for _, class_id in self.detection_history:
            counts[class_id] += 1
        self.current_counts = counts.copy()
        warning_count = counts[2]
        caution_count = counts[0]
        abnormal_count = warning_count + caution_count
        abnormal_ratio = abnormal_count / total_detections
        new_alert_message = ""
        if abnormal_ratio >= self.DANGER_RATIO_THRESHOLD:
            if warning_count >= caution_count:
                temp_message = f"🚨 경고 발생: 비정상 비율 {abnormal_ratio:.0%} (Warning 우세)"
                if (current_time - self.last_alert_time['warning']) >= self.ALERT_COOLDOWN_SECONDS:
                    print(f"🚨🚨🚨 [경고 발생] {temp_message} 🚨🚨🚨")
                    self.last_alert_time['warning'] = current_time
                    new_alert_message = "🚨🚨🚨 [경고 발생] " + temp_message + " 🚨🚨🚨"
                else:
                    new_alert_message = self.current_alert_message
            else:
                temp_message = f"⚠️ 주의 발생: 비정상 비율 {abnormal_ratio:.0%} (Caution 우세)"
                if (current_time - self.last_alert_time['caution']) >= self.ALERT_COOLDOWN_SECONDS:
                    print(f"⚠️⚠️⚠️ [주의 발생] {temp_message} ⚠️⚠️⚠️")
                    self.last_alert_time['caution'] = current_time
                    new_alert_message = "⚠️⚠️⚠️ [주의 발생] " + temp_message + " ⚠️⚠️⚠️"
                else:
                    new_alert_message = self.current_alert_message
        else:
            new_alert_message = ""
        if new_alert_message != self._last_displayed_alert:
            self.current_alert_message = new_alert_message
            self._last_displayed_alert = new_alert_message

    def _capture_loop(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue
            if self.capture_q.full():
                try: self.capture_q.get_nowait()
                except queue.Empty: pass
            self.capture_q.put(frame)

    def _preprocess_loop(self):
        while not self.stop_event.is_set():
            try:
                frame = self.capture_q.get(timeout=0.1)
            except queue.Empty:
                continue
            original_h, original_w = frame.shape[:2]
            resized_frame, r, dw, dh = letterbox(frame, (self.input_width, self.input_height))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            input_data = (rgb_frame.astype(np.float32) / 255.0).transpose(2, 0, 1)
            self.preprocess_q.put((frame, input_data, original_w, original_h, r, dw, dh))

    def _infer_loop(self):
        while not self.stop_event.is_set():
            try:
                frame, input_data, original_w, original_h, r, dw, dh = self.preprocess_q.get(timeout=0.1)
            except queue.Empty:
                continue
            self.cuda_context.push()
            try:
                np.copyto(self.h_input, np.ascontiguousarray(input_data).ravel())
                stream = cuda.Stream()
                push_range("GPU_Inference")
                try:
                    cuda.memcpy_htod_async(self.d_input, self.h_input, stream)
                    self.context.execute_async_v3(stream_handle=stream.handle)
                    cuda.memcpy_dtoh_async(self.h_output, self.d_output, stream)
                    stream.synchronize()
                finally:
                    pop_range()
                predictions = self.h_output.reshape(self.output_shape)[0].transpose(1, 0)
                self.infer_q.put((frame, predictions, original_w, original_h, r, dw, dh))
            except Exception as e:
                print(f"Infer error: {e}")
            finally:
                self.cuda_context.pop()

    def _postprocess_loop(self):
        while not self.stop_event.is_set():
            try:
                frame, predictions, original_w, original_h, r, dw, dh = self.infer_q.get(timeout=0.1)
            except queue.Empty:
                continue
            annotated_frame = self.postprocess(frame, predictions, original_w, original_h, r, dw, dh)
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            if self.output_q.full():
                try: self.output_q.get_nowait()
                except queue.Empty: pass
            with self.result_lock:
                self.output_q.put(jpeg.tobytes())
                self.latest_result = jpeg.tobytes()

    def get_frame(self):
        with self.result_lock:
            return self.latest_result

    def release_resources(self):
        if not self.is_running:
            print("ℹ️ release_resources가 이미 호출되었거나 초기화되지 않았습니다. 중복 실행 방지.")
            return
        print("🛑 자원 해제 시작...")
        self.is_running = False
        self.stop_event.set()
        try:
            self.cuda_context.push()
            if self.cap and self.cap.isOpened():
                self.cap.release()
                print("✅ 카메라 해제 완료.")
            if self.context: del self.context
            if self.engine: del self.engine
            if self.d_input: self.d_input.free()
            if self.d_output: self.d_output.free()
            self.h_input = None
            self.h_output = None
            self.context = None
            self.engine = None
            self.d_input = None
            self.d_output = None
        except Exception as e:
            print(f"🚨 자원 해제 중 오류 발생: {e}")
        finally:
            self.cuda_context.pop()
            print("✅ 모든 자원 해제 완료. 프로세스 종료 준비.")

    def __del__(self):
        print("ℹ️ VideoCamera 객체가 소멸됩니다. 명시적으로 release_resources를 호출했는지 확인하세요.")
        if self.is_running:
            self.release_resources()