# 🔥 Thermal Fall Detection System (Portfolio Version)

본 프로젝트는 **실시간 열화상 낙상 감지 시스템**의 포트폴리오 요약입니다.  
TensorRT 기반 YOLO 모델을 활용하여 낙상 및 위험 상황을 감지하며, PyCUDA 및 GStreamer를 통해 **RTSP 스트리밍 기반 초고속 추론**을 실현합니다.

---

## 🧠 프로젝트 개요
- 열화상 카메라(RTSP)로부터 실시간 영상 수신
- TensorRT YOLO 엔진을 통한 실시간 추론
- OpenCV + PyCUDA 기반 후처리
- 알림 트리거 기능 (낙상 감지율 기반)

---

## 🖥️ 기술 스택

| 구성 요소     | 사용 기술                         |
|--------------|----------------------------------|
| 추론 엔진     | TensorRT (.engine, FP16)         |
| 전처리        | OpenCV, Letterbox                |
| RTSP 수신     | GStreamer (rtspsrc)              |
| CUDA 관리     | PyCUDA                           |
| 병렬 처리     | Python Threading + Queue         |
| 프레임 출력   | OpenCV + JPEG 인코딩             |
| 알림 관리     | 감지율 기반 위험도 판단 시스템   |

---

## 🧰 주요 구성 파일 (민감 로직 제외)
- `infer_template.py`: 추론 구조 템플릿 (TensorRT 연동 구조 설명 위주)
- `model_conversion.md`: YOLO → ONNX → TensorRT 변환 절차 정리
- `deploy_guide.md`: GPU 설정, 가상환경 구성, GStreamer 설치 가이드
- `app_structure.md`: Flask 기반 실시간 서버 구조 설명

⚠️ 본 레포지토리는 포트폴리오 목적의 요약 버전입니다. 실제 엔진 파일, 민감 코드, 데이터는 포함되어 있지 않습니다.

---

## 📸 시스템 구성도 & 결과

| 구성도 | 결과 화면 |
|--------|------------|
| ![구성도](system_diagram.png) | ![결과](screenshots/detection_ui.png) |

---

## 👥 Contributors

### Hyunmin Bang (방현민)
- 📧 banghyunmin1999@gmail.com  
- 🔗 [GitHub](https://github.com/banghyunmin1999)  
- 🔗 [LinkedIn](https://www.linkedin.com/in/hyunmin-bang-6b944936b)  
- 🤗 [Hugging Face](https://huggingface.co/banghyunmin)

### Eunji Choi (최은지)
- 📧 creweunji@gmail.com  
- 🔗 [GitHub](https://github.com/Eunji-Choi-Lulu)  
- 🔗 [LinkedIn](https://www.linkedin.com/in/eunji-choi-b3bbb788)  
- 🤗 [Hugging Face](https://huggingface.co/EunjiChoi)

---

## 📥 발표 자료
- 🔗 [슬라이드 다운로드 (PDF)](링크삽입예정)

---

## 📝 기여 내용 정리 예시 (README에 직접 포함하거나 `CONTRIBUTING.md`에 별도로 추가)

```md
## 📌 내 기여 내용 (방현민)

- YOLOv11 기반 열화상 낙상 감지 모델 학습 및 성능 검증
- ONNX → TensorRT 변환 자동화 스크립트 구현 (FP16/INT8)
- GStreamer + PyCUDA 기반 실시간 추론 파이프라인 최적화
- 후처리 병목 제거, 평균 처리시간 35ms 달성
- Flask 기반 추론 서버 구조 및 RTSP 연동
