# Windows 초기 설정

## Step to run code
### 저장소 복제
```
git clone https://github.com/ziho9593/yolov7-pose-tracking.git
```
### 폴더 이동
```
cd yolo7-pose-tracking
```
### 가상환경 생성 & 실행 - Pipenv 사용
```
pip install --upgrade pip
pip install pipenv
pipenv --python 3.X
pipenv shell
```
### 패키지 설치
```
pipenv install
```
### (GPU를 사용할 경우) CUDA 설치
- [링크](https://pytorch.org/get-started/previous-versions/)에서 사양에 맞는 torch와 CUDA 버전을 선택하여 pip 명령을 복사
- pipenv install은 --extra--index-url 옵션이 동작하지 않기 때문에, 복사한 pip 명령 앞에 pipenv run을 붙여 다운로드 실행
```
# 예시
pipenv run pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```