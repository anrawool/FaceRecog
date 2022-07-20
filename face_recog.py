import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical


class FaceRecogTrainer():
    pass
    # Functions:
    # 1. Train, Test Set Preprocesser

    def trainTestPP(self, Traindirectory, TestDirectory, target_size=(64, 64), batch_size=32, network_type='binary', rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True):
        train_datagen = ImageDataGenerator(
            rescale=rescale,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip)  # Parameters for Datapreprocessing

        self.training_set = train_datagen.flow_from_directory(
            Traindirectory,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=network_type)  # Training Set

        test_datagen = ImageDataGenerator(rescale=1./255)
        self.test_set = test_datagen.flow_from_directory(
            TestDirectory,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=network_type)  # Test Set
        # 2. Model Creator

    def ModelCreator(self, activation='sigmoid', out_neurons=1, input_shape=[64, 64, 3]):
        # Building The CNN
        if len(self.training_set.class_indices) > 2:
            out_neurons = len(self.training_set.class_indices)
        else:
            out_neurons = out_neurons
        self.cnn = tf.keras.models.Sequential()

        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,
                                            activation='relu', input_shape=input_shape))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        self.cnn.add(tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, activation='relu'))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        self.cnn.add(tf.keras.layers.Flatten())
        self.cnn.add(tf.keras.layers.Dense(units=140, activation='relu'))
        self.cnn.add(tf.keras.layers.Dense(
            units=out_neurons, activation=activation))

    # 3. Model Compiler

    def ModelCompiler(self, optim="adam", loss='binary_crossentropy', metrics=['accuracy']):
        # # Compiling the CNN
        self.cnn.compile(optimizer=optim, loss=loss,
                         metrics=metrics)

    # 4. Model Trainer

    def ModelTrainer(self, epochs=5):
        self.cnn.fit(x=self.training_set,
                     validation_data=self.test_set, epochs=epochs)

    # 5. Model Saver

    def ModelSaver(self, saveDirectory):
        self.cnn.save(f'{saveDirectory}/model')

    def TestImagePred(self, pred_img_path, model_path, target_size=(64, 64)):
        cnn = tf.keras.models.load_model(model_path)
        Testimage = image.load_img(
            pred_img_path, target_size=target_size)
        Testimage = image.img_to_array(Testimage)
        Testimage = np.expand_dims(Testimage, axis=0)
        result = cnn.predict(Testimage/255.0)
        return result


# Example Usuage

FaceRecog = FaceRecogTrainer()
FaceRecog.trainTestPP('./Dataset/Training_Set',
                      './Dataset/Test_Set', network_type='categorical')  # Training and Testing Dataset Preprocessor
FaceRecog.ModelCreator(activation='softmax')  # Model Creation
FaceRecog.ModelCompiler(loss='categorical_crossentropy')  # Model Compiler
FaceRecog.ModelTrainer(epochs=1)  # Model Trainer
FaceRecog.ModelSaver(saveDirectory="test_model")  # Model Saver
result = FaceRecog.TestImagePred('single_pred/image.jpeg',
                                 'saved_model/my_model', target_size=(64, 64))  # Model Tester

results = []

for i in result[0]:
    x = round(i, 2)
    results.append(x)
print(results)
