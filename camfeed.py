import os
import cv2


cap = cv2.VideoCapture(0)
result, image = cap.read()

if result:

    # showing result, it take frame name and image
    # output
    cv2.imshow("Camfeed", image)

    # saving image in local storage
    cv2.imwrite("single_pred/camfeed.png",
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # If keyboard interrupt occurs, destroy image
    # window
    cv2.waitKey(5)
    cv2.destroyAllWindows()

# print("Starting Recognition")
# os.system("python test.py")

# If captured image is corrupted, moving to else part
