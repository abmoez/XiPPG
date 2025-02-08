import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from model import Xception
import argparse
from tensorflow.keras.optimizers import Adam    
from tensorflow.keras.metrics import RootMeanSquaredError

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
from Video_Generator import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xception model")
    parser.add_argument("--input_shape", type=int, nargs="+", default=[50, 120, 160, 3], help="Input shape")
    #parser.add_argument("--num_classes", type=int, default=1, help="Number of classes")
    #parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    #parser.add_argument("--l1", type=float, default=0.001, help="L1 regularization")
    #parser.add_argument("--l2", type=float, default=0.001, help="L2 regularization")

    args = parser.parse_args()

    #model = Xception(input_shape=args.input_shape, num_classes=args.num_classes, dropout_rate=args.dropout_rate, l1=args.l1, l2=args.l2)
    model = Xception(input_shape=args.input_shape)

    #model.summary()

    opt = Adam(learning_rate=0.0001, weight_decay=0.5)

    rmse = RootMeanSquaredError()
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae', rmse])



    # Every epoch: check 'val_root_mean_squared_error' and save the best weights
    checkpoint_2 = ModelCheckpoint(
    'weights-{epoch:02d}-{val_root_mean_squared_error:.4f}.weights.h5',  # <- must end in `.weights.h5`
    monitor='val_root_mean_squared_error',
    verbose=1,
    save_best_only=True,
    save_weights_only=True
)


# Every 10 epochs: save weights (based on 'val_mean_absolute_error')
    checkpoint = ModelCheckpoint(
    'weights.{epoch:02d}-{val_root_mean_squared_error:.4f}.weights.h5',
    monitor='val_mean_absolute_error',
    verbose=10,
    save_best_only=True,
    save_weights_only=True  # <--- MUST be True for .h5
    # Optionally specify save_freq or period for custom saving intervals
)
    history_checkpoint = CSVLogger("history.csv", append=False)

    # use tensorboard can watch the change in time
    tensorboard_ = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    early_stopping = EarlyStopping(monitor='val_root_mean_squared_error', patience=5, verbose=1, mode='auto')

    """
    if (CONTINUE_TRAINING == True):
        history = pd.read_csv('history.csv')
        INITIAL_EPOCH = history.shape[0]
        model.load_weights('weights_%02d.h5' % INITIAL_EPOCH)
        checkpoint_2.best = np.min(history['val_root_mean_squared_error'])
    else:
        INITIAL_EPOCH = 0
    """

    datagen = ImageDataGenerator()

    batch_size_train = 1
    batch_size_test = 1
    
    
    # we need to change the directory to the path where the data is stored
    # out data must be transformed to the format that the custom dataget can understand
    # the data should be in the following format:
    # THIS DATA WILL BE USED TO TRAIN THE MODEL DIRECTLY, SO RAW DATA MUST BE SEGMENTED INTO FRAMES OF FACES FIRST
    
    """"
    dataset/                            # dataset root should be *mirrored* into video_path and heart_rate_path
    ├── videos_path/
    │   ├── subject_01/
    │   │   ├── video_01/               # video folder contains 50 frames per video as 2s of 25 fps video
    │   │   │   ├── frame_0001.jpg
    │   │   │   ├── frame_0002.jpg
    │   │   │   ├── ...
    │   │   ├── video_02/
    │   │   │   ├── frame_0001.jpg
    │   │   │   ├── frame_0002.jpg
    │   │   │   ├── ...
    │   │   └── ...
    │   ├── subject_02/
    │   │   ├── video_01/
    │   │   │   ├── frame_0001.jpg
    │   │   │   ├── frame_0002.jpg
    │   │   │   ├── ...
    │   │   ├── video_02/
    │   │   │   ├── frame_0001.jpg
    │   │   │   ├── ...
    │   │   └── ...
    │   └── ...
    ├── heart_rate_path/
    │   ├── subject_01/
    │   │   ├── video_01/
    │   │   │   ├── Pu_heart_rate.csv   # heart rate file should start with Pu_ and be in csv format
    │   │   ├── video_02/
    │   │   │   ├── Pu_heart_rate.csv
    │   │   └── ...
    │   ├── subject_02/
    │   │   ├── video_01/
    │   │   │   ├── Pu_heart_rate.csv
    │   │   ├── video_02/
    │   │   │   ├── Pu_heart_rate.csv
    │   │   └── ...
    │   └── ...

    """

    # In terms of labels, the heart rate, it should be in csv format for each video
    # each csv file should have the following format:
    # Number of HR samples should be equal to the number of frames in the video
    # The code expects multiple values
    # If each CSV contains only one value (currently our case), it won’t align with the frames in the video as the code expects

    train_data = datagen.flow_from_directory(directory='dataset/videos_path',
                                            label_dir='dataset/heart_rate_path',
                                            target_size=(120, 160), class_mode='label', batch_size=4,
                                            frames_per_step=50, shuffle=False)

    test_data = datagen.flow_from_directory(directory='dataset/videos_path',
                                            label_dir='dataset/heart_rate_path',
                                            target_size=(120, 160), class_mode='label', batch_size=4,
                                            frames_per_step=50, shuffle=False)

    history = model.fit(train_data, epochs=3,
                        steps_per_epoch=len(train_data.filenames) // 3200,
                        validation_data=test_data, validation_steps=len(test_data.filenames) // 3200,
                        callbacks=[history_checkpoint, checkpoint_2])



    values = history.history
    validation_loss = values['val_loss']
    validation_mae = values['val_mae']
    training_mae = values['mae']
    validation_rmse = values['val_root_mean_squared_error']
    training_rmse = values['root_mean_squared_error']
    training_loss = values['loss']

    epochs = range(100)

    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.title('Epochs vs Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, training_mae, label='Training MAE')
    plt.plot(epochs, validation_mae, label='Validation MAE')
    plt.title('Epochs vs MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

    plt.plot(epochs, training_rmse, label='Training RMSE')
    plt.plot(epochs, validation_rmse, label='Validation RMSE')
    plt.title('Epochs vs RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()
    plt.show()


