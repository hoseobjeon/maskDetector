# 텐서플로우 마스크 판별기

```
카메라를 이용해서 실시간으로 얼굴을 파악해서 마스크를 썼는지 감지한다.
```

![image](https://raw.githubusercontent.com/hoseobjeon/portfolio/master/assets/img/maskDetector.png)

## 사용 라이브러리
* keras, cv2, os, numpy, pickle, sklearn, winsound, datatime

### 1. 이미지 준비
> 마스크, 노마스크 두 종류의 이미지들을 os로 불러서 opencv로 이미지 사이즈 변경 및 정규화를 통해 과적합 방지 그리고 그 이미지들을 두 종류로 분류한 후 np를 사용해서 pickle에 파일로 저장

### 2. 텐서플로우 모델
> pickle로 저장된 파일 불러와서 CNN 모델 두번 사용 > 플레튼으로 일정하게 만든 후 > 과적합 방지를 위해 50% 드롭아웃 > 계층 50개 > softmax 사용해서 계층 2개 (마스크 / 노마스크) 훈련할 파일과 검사할 파일은 8:2 비율로 두었고 10번 반복 학습

### 3. 비디오 판별기
> haar cascade 알고리즘을 사용해서 얼굴 감지, 마스크 사용 유무에 따른 메세지, 마스크 썼을때 소리, 스크린샷 기능
