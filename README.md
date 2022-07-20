# Face Recognition Module

This is a free trainable face recognition module anybody can use. You can either use the Camfeed of just a photo.

## Using the Module

The Module is divided in one class called FaceRecognition, there are definitions that help you to do different tasks. You can play around with the images and the neural network layers.

### Creating the dataset

1. Create a Directory named 'Dataset' if it already doesn't exist
2. Then create two directories named 'Test_Set' and 'Training_Set', if you want to name them something else, you will have to provide the path while training the neural network
3. Inside the 'Training_Set' folder, create teh diffrent category folders you want. For example, if the classification is of cats and dogs, create two folders named 'Cats' and 'Dogs'
4. Place respective photos in their respective folders. If you do not have enough photos, you can use the duplicate_dataset definition give in DatasetGeneralizer.py file to duplicate images. It is recommended to have at least 10 unique photos.
5. Train the neural network, which can take as long as 5 minutes or upto 1 hour, depending on the number of categories it has to train on and the hardware.

If the person using this module is fimiliar with neural networks and Tensorflow, they will know how to tweak the neural network to fit their requirements.
Thank You, hope you like this module.
