import os
import signal
import threading
import time
from flask import Flask, render_template, Response, request
from camera import VideoCamera
import json
# cv2 import는 더 이상 app.py에 필요 없습니다.

app = Flask(__name__)
camera = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

# [핵심 수정] camera.py가 인코딩까지 완료하므로, gen()은 데이터를 전달만 합니다.
def gen(camera_instance):
    """웹으로 스트리밍할 프레임을 생성하는 제너레이터 함수"""
    while True:
        # camera.py로부터 최종 처리 및 인코딩된 JPEG 바이트를 받아옵니다.
        frame_bytes = camera_instance.get_frame()
        
        if frame_bytes is None:
            time.sleep(0.01) # 아주 잠깐 대기
            continue

        # HTTP 스트리밍 형식에 맞게 헤더와 JPEG 바이트를 합쳐서 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status_feed')
def status_feed():
    def status_generator():
        while True:
            # camera 인스턴스의 최신 상태를 직접 읽어옵니다.
            counts = camera.current_counts
            message = camera.current_alert_message
            json_data = json.dumps({
                'counts': counts,
                'message': message,
                'timestamp': time.time()
            })
            yield f"data: {json_data}\n\n"
            time.sleep(1)
    return Response(status_generator(), mimetype='text/event-stream')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    def shutdown_server():
        time.sleep(0.5)
        try:
            camera.release_resources()
        except Exception as e:
            print(f"자원 해제 중 오류: {e}")
        os.kill(os.getpid(), signal.SIGINT)
    threading.Thread(target=shutdown_server).start()
    return '서버가 안전하게 종료됩니다.'
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True, use_reloader=False)