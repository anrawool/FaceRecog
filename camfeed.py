import cv2
import os
from PIL import Image
name = input("Enter you name: ")
num_images = int(input("Enter the number of images required: "))
def create_images(name):
    os.chdir(f"/Users/abhijitrawool/Documents/Sarthak/Programming_Projects/Face Recog/Dataset/Training_Set/Sarthak/")
    for i in range(1046, num_images+1):
        print(i)
        cam_port = 0
        cam = cv2.VideoCapture(cam_port)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        result, image = cam.read()
        if result:
            cv2.imwrite(f"{name.lower()}_{i}.jpg", image)
        else:
            print("No image detected. Please! try again")
create_images(name)

