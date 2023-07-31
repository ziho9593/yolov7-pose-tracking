# yolov7-pose-tracking
YOLOv7 Object Tracking + Pose Estimation


### 설명
YOLOv7에 기반한 Object Tracking과 YOLO-Pose의 Pose Estimation을 사용하여 사람의 움직임을 분석하기 위한 시계열 데이터를 추출하는 것에 활용


### 초기 설정
- [Windows](https://github.com/ziho9593/yolov7-pose-tracking/blob/pose/docs/windows.md)


### 사전학습 모델 준비
- [yolov7-w6-pose](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)를 직접 다운로드하여 작업 폴더 안으로 이동
- [yolov7.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt)의 경우 작업 폴더 내에 존재하지 않으면 자동으로 다운로드


### 기본 사용법
```
# Basic Usage
python tracker.py --source "your video.mp4"

# for Using GPU (CUDA Installation Required)
python tracker.py --device 0 --source "your video.mp4"

# for Saving Keypoints
python tracker.py --save-kpts --source "your video.mp4"

# for Specific Class (0: Person)
python tracker.py --classes 0 --source "your video.mp4"

# for WebCam
python tracker.py --source 0

# for External Camera
python tracker.py --source 1

# for LiveStream (IP Stream URL Format i.e "rtsp://username:pass@ipaddress:portno/video/video.amp")
python tracker.py --source "your IP Camera Stream URL" --device 0
```


### 결과 예시
<p><img src='./docs/src/ohtani.gif' height=400></p>


### 참고자료
- https://arxiv.org/ftp/arxiv/papers/2204/2204.06806.pdf
- https://github.com/WongKinYiu/yolov7/tree/pose
- https://github.com/RizwanMunawar/yolov7-pose-estimation
- https://github.com/RizwanMunawar/yolov7-object-tracking