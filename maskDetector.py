### 캠으로 마스크 판단기

### 라이브러리 가져오기
import cv2
import os
import numpy as np
import pickle
import winsound
import datetime
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D,MaxPooling2D
from sklearn.model_selection import train_test_split


### 데이터 준비하기

# 데이터 리스트로 넣어두기
data_directroy = "dataset" # 트레이닝할 데이터 폴더
classes = os.listdir(data_directroy)
labels = [i for i in range(len(classes))]
label_dict = dict(zip(classes, labels))

img_size = 100 # 이미지 사이즈

training_data = [] # 이미지 데이터
training_class = [] # 이미지 번호

def create_training_data():
    for category in classes:
        data_path = os.path.join(data_directroy, category)
        class_num = classes.index(category)
        images_path = os.listdir(data_path)

        for img in os.listdir(data_path):
            img_path = os.path.join(data_path, img)
            img_array = cv2.imread(img_path)

            try:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                new_array = cv2.resize(gray, (img_size, img_size))
                training_data.append(new_array)
                training_class.append(label_dict[category])
                            
            except Exception as e:
                print('Exception', e)

create_training_data()

# 사이즈 변환
training_data = np.array(training_data) / 255.0 # 정규화해야 오버라이딩 안됨
training_data = np.reshape(training_data, (-1, img_size, img_size, 1)) # 3차원 > 4차원

training_class = np.array(training_class)
training_class = np_utils.to_categorical(training_class) # 클래스가 지정된 데이터 배열로 변환

# 리스트로 저장하기
# pickle을 통해서 입력하기
pickle_out = open("training_data.pickle", 'wb')
pickle.dump(training_data, pickle_out)
pickle_out.close()
pickle_out = open("training_class.pickle", 'wb')
pickle.dump(training_class, pickle_out)
pickle_out.close()



### 텐서플로우 모델 만들기

# pickle을 통해서 불러오기
pickle_in = open("training_data.pickle", "rb")
training_data = pickle.load(pickle_in)
pickle_in = open("training_class.pickle", "rb")
training_class = pickle.load(pickle_in)

# 순차모델
model = Sequential()

# 200 커널 사이즈 3 x 3 설정
model.add(Conv2D(200,(3,3),input_shape=training_data.shape[1:]))
# 복잡도를 낮춤, 0보다 작으면 0 그렇지 않으면 값에 비례하여 증가
model.add(Activation('relu'))
# 과적합 방지 2,2 사이즈를 사용해서 사이즈를 반으로 줄임
model.add(MaxPooling2D(pool_size=(2,2)))

# 200 커널 사이즈 3 x 3 설정
model.add(Conv2D(100,(3,3)))
# 복잡도를 낮춤, 0보다 작으면 0 그렇지 않으면 값에 비례하여 증가
model.add(Activation('relu'))
# 과적합 방지 2,2 사이즈를 사용해서 사이즈를 반으로 줄임
model.add(MaxPooling2D(pool_size=(2,2)))

# 모델내에서 reshape의 기능
model.add(Flatten())
# 오버피팅 방지, 50% 드롭아웃
model.add(Dropout(0.5))
# 50 계층 설정
model.add(Dense(50, activation='relu'))
# 마스크 / 노 마스크 두가지 카테고리
model.add(Dense(2, activation='softmax'))

# categorical_crossentropy로 구분 (마스크 / 노 마스크), adam 사용 softmax에 적합 계층이 경사 크기에 영향을 크게 받지 않음
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 테스트 비율 지정
train_data, valid_data, train_class, valid_class = train_test_split(training_data, training_class, test_size=0.2, shuffle=None)
# fit 지정 10번 반복, 검사 20퍼
model.fit(train_data, train_class, epochs=10, validation_split=0.2)
# 테스트 파일로 확인
print(model.evaluate(valid_data, valid_class))

### 모델 저장
model.save('model.h5')



### 비디오 판별기 (소리추가)

# 마스크 감지됬을때 소리 설정
frequency = 1000 # 주파수 1000 헤르츠로 설정
duration = 100 # 0.1초에 한번씩

# 모델 불러오기
model = load_model('model.h5')

# 비디오가 안됬을때 > 내장 캠이 안됬을때 > 외장캠으로 열기
video_file = "mask_video.mp4"
cam = cv2.VideoCapture(video_file)
if not cam.isOpened():
    cam = cv2.VideoCapture(0)
elif not cam.isOpened():
    cam = cv2.VideoCapture(1)
elif not cam.isOpened():
    raise IOError("Cannot open webcam")

# 오브젝트 검출 알고리즘 분류할때 사용
facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict={0:'mask', 1:'no_mask'}
# 마스크 초록색G, 노 마스크 레드R (BGR 순서)
color_dict={0:(0, 255, 0), 1:(0, 0, 255)}

while True:

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 얼굴 위치 좌표 반환: 이미지, 1.3 크기(인식기회 증가, 속도 감소), 이웃수 5(품질증가, 검출개수 감소)   
    faces = facecascade.detectMultiScale(gray, 1.3, 5)  

    for x, y, w, h in faces:
        face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)

        # 각행에 높은 인덱스값
        label = np.argmax(result, axis=1)[0]

        # 좌표, 색, 두께
        cv2.rectangle(img, (x,y) , (x+w,y+h), color_dict[label], 2)

        if labels_dict[label] == 'mask':
            status = "Good you're wearing a mask"
            x1, y1, w1, h1 = 0, 0, 400, 75
            # 표시화면
            cv2.rectangle(img, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
            # 표시글
            cv2.putText(img, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # 마스크 썼을때 소리
            winsound.Beep(frequency, duration)

        elif labels_dict[label] == 'no_mask':
            status = "Please wear a mask"
            x1, y1, w1, h1 = 0, 0, 280, 75
            # 표시화면
            cv2.rectangle(img, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
            # 표시글
            cv2.putText(img, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Mask Detector', img)
    key = cv2.waitKey(1)
    if key == ord('q'): # q로 멈춤
        break
    elif key == ord('s'): # s 사진 저장
        file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") + '.png'
        cv2.imwrite(file, img)
        print(file, ' saved')

cv2.destroyAllWindows()
cam.release()