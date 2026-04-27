import cv2

# 보통 /dev/video0이 기본 카메라입니다.
cap = cv2.VideoCapture("/dev/video0")

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('WSL2 Camera Test', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()