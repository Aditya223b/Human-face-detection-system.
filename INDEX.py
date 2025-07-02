import cv2 
import imutils 
# Load the pre-trained Haar Cascade classifier for gun detection
gun_cascade = cv2.CascadeClassifier('cascade.xml')
camera = cv2.VideoCapture(0)
firstFrame = None 
gun_exists = None
while True: 
    ret, frame = camera.read()
    frame = imutils.resize(frame, width = 500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gun = gun_cascade.detectMultiScale(gray, 
                                       1.3,5,
                                       minSize = (100, 100))
    if len(gun)>0:
        gun_exists = True

    for (x, y, w, h) in gun:
        frame = cv2.rectangle(frame, (x+w, y+h), (x, y), (0, 255, 0), 2)
    
    rio_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]

    if firstFrame is None:
        firstFrame = gray
        continue

    cv2.imshow("security feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if gun_exists:
        print("Gun detected!")
    else:
        print("No gun detected.")

camera.release()
cv2.destroyAllWindows()