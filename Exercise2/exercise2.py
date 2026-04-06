import cv2

# 1. OpenCV에서 기본 제공하는 얼굴 인식 인공지능 모델(Haar Cascade) 불러오기
# (cv2.data.haarcascades를 쓰면 복잡한 xml 파일을 따로 다운받지 않아도 됩니다!)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. 이미지 불러오기 (여기에 본인의 사진 파일 이름을 넣으세요)
img = cv2.imread('sample.jpg')

# 이미지를 흑백으로 변환 (얼굴 인식은 흑백 상태에서 훨씬 빠르고 정확하게 작동합니다)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. 얼굴 검출 실행!
# scaleFactor: 이미지를 얼마나 축소하며 찾을지 (보통 1.1)
# minNeighbors: 얼굴 주변에 얼마나 많은 사각형이 겹쳐야 진짜 얼굴로 인정할지 (보통 4~6)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 4. 찾은 얼굴에 네모 상자(박스) 그리기
# x, y는 좌측 상단 좌표, w, h는 가로/세로 길이
for (x, y, w, h) in faces:
    # img에, (x,y)부터 (x+w, y+h)까지, 파란색(255,0,0)으로, 두께 2의 선을 그림
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 5. 결과 화면 띄우기
print(f"총 {len(faces)}개의 얼굴을 찾았습니다!")
cv2.imshow('Face Detection Result', img)

# 아무 키나 누르면 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()