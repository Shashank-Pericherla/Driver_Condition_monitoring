import cv2
import time
from matplotlib import pyplot as plt
from deepface import DeepFace


TIMER = int(5)

cap = cv2.VideoCapture(0)

while True:

    ret, img = cap.read()
    cv2.imshow('a', img)

    k = cv2.waitKey(125)


    if k == ord('q'):
        prev = time.time()

        while TIMER >= 0:
            ret, img = cap.read()


            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(TIMER),
                        (200, 250), font,
                        7, (0, 255, 255),
                        4, cv2.LINE_AA)
            cv2.imshow('a', img)
            cv2.waitKey(125)

            cur = time.time()


            if cur - prev >= 1:
                prev = cur
                TIMER = TIMER - 1

        else:
            ret, img = cap.read()

            cv2.imshow('a', img)

            cv2.waitKey(2000)



    elif k == 27:
        break

cap.release()

cv2.destroyAllWindows()




gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale', gray_image)
cv2.waitKey(0)
filename = "88.jpg"

cv2.destroyAllWindows()



dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)

plt.subplot(121), plt.imshow(gray_image)
plt.subplot(122), plt.imshow(dst)
plt.show()




result = DeepFace.analyze(dst, actions=['emotion'])

a = result['dominant_emotion']
print(a)

unstable_emotions=["angry", "fear", "sad", "suprise"]

if a in unstable_emotions:
    img = cv2.imread("caution.jpg", cv2.IMREAD_COLOR)
    cv2.imshow("CAUTION", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
