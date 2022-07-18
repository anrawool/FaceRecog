import os
import cv2


cap = cv2.VideoCapture(0)
result, image = cap.read()

if result:
    cv2.imshow("Camfeed", image)

    cv2.imwrite("single_pred/camfeed.png",
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    cv2.waitKey(5)
    cv2.destroyAllWindows()
