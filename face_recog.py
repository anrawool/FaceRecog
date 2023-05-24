import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class FaceRecogTrainer():
    pass
    # Functions:
    # 1. Train, Test Set Preprocesser

    def trainTestPP(self, Traindirectory, TestDirectory, target_size=(64, 64), batch_size=128, network_type='binary', rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True):
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
        self.cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', input_shape=input_shape))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        self.cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        self.cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        self.cnn.add(tf.keras.layers.Flatten())
        self.cnn.add(tf.keras.layers.BatchNormalization())
        self.cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
        self.cnn.add(tf.keras.layers.Dropout(0.2))
        self.cnn.add(tf.keras.layers.BatchNormalization())
        self.cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
        self.cnn.add(tf.keras.layers.Dropout(0.2))
        self.cnn.add(tf.keras.layers.BatchNormalization())
        self.cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))
        self.cnn.add(tf.keras.layers.BatchNormalization())
        self.cnn.add(tf.keras.layers.Dense(units=32, activation='relu'))
        self.cnn.add(tf.keras.layers.Dropout(0.2))
        self.cnn.add(tf.keras.layers.BatchNormalization())
        self.cnn.add(tf.keras.layers.Dense(units=16, activation='relu'))
        self.cnn.add(tf.keras.layers.Dropout(0.2))
        self.cnn.add(tf.keras.layers.BatchNormalization())
        self.cnn.add(tf.keras.layers.Dense(units=8, activation='relu'))
        self.cnn.add(tf.keras.layers.Dense(units=out_neurons, activation=activation))
        return self.cnn

    # 3. Model Compiler

    def ModelCompiler(self, optim="adam", loss='binary_crossentropy', metrics=['accuracy']):
        # # Compiling the CNN
        self.cnn.compile(optimizer=Adam(learning_rate=0.01), loss=loss,
                         metrics=metrics)

    # 4. Model Trainer

    def ModelTrainer(self, epochs=5):
        history = self.cnn.fit(x=self.training_set,
                     validation_data=self.test_set, epochs=epochs, callbacks=[EarlyStopping(monitor='val_accuracy', patience=10, min_delta=0.001, restore_best_weights=True)])
        return history

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
# FaceRecog.trainTestPP('./Dataset/Training_Set',
#                    './Dataset/Test_Set', network_type='categorical')  # Training and Testing Dataset Preprocessor
# model = FaceRecog.ModelCreator(activation='softmax')  # Model Creation
# print(model.summary())
# FaceRecog.ModelCompiler(loss='categorical_crossentropy')  # Model Compiler
# FaceRecog.ModelTrainer(epochs=10)  # Model Trainer
# FaceRecog.ModelSaver(saveDirectory="test_model_latest")  # Model Saver
results = []
for i in range(1, 10):
    # result = FaceRecog.TestImagePred(f'Dataset/Test_Set/sarthak/sarthak_{i}.jpg', 'test_model_latest/model', target_size=(64, 64))  # Model Tester
    result = FaceRecog.TestImagePred(f'Dataset/Test_Set/Sarthak/sarthak_{i}.jpg', 'test_model_latest/model', target_size=(64, 64))  # Model Tester
    for num in result[0]:
        num = round(num, 2)
    results.append(result[0])
final_res = []
for res in results:
    res_nums = list(res)
    res = []
    for num in res_nums:
        num = round(num, 2)
        res.append(num)
    res_nums.append(res_nums)

def check_results(list_of_results, index):
    correct_pred = 0
    incorrect_pred = 0
    for list_result in list_of_results:
        if max(list_result) == list_result[index-1]:
            correct_pred += 1
        else:
            incorrect_pred += 1
    total_pred = incorrect_pred + correct_pred
    return {"Total:": total_pred, "Correct:": correct_pred, "Incorrect:": incorrect_pred}

final_results_checked = check_results(results, 4)
print(results)
print(final_results_checked)
