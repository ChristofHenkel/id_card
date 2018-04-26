from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

BATCH_SIZE = 16

train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                    vertical_flip=True,
                                    horizontal_flip=True,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    zoom_range=0.2,
                                    shear_range=0.2,

                                    )

train_generator = train_data_gen.flow_from_directory(directory='assets/train/',
                                                     target_size=(224, 224),
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='categorical')

valid_data_gen = ImageDataGenerator(rescale=1. / 255)
valid_generator = valid_data_gen.flow_from_directory(directory='assets/valid/',
                                                     target_size=(224, 224),
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='categorical')

model = load_model('model.hdf5')
K.set_value(model.optimizer.lr,0.0001)


check_point = ModelCheckpoint('model_retrained.hdf5', monitor="val_loss", mode="min", save_best_only=True, verbose=1)
early_stop = EarlyStopping(patience=4)
rop = ReduceLROnPlateau(patience=2, factor=0.1, min_lr=0)

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.classes.size // BATCH_SIZE,
                              epochs=30,
                              validation_data=valid_generator,
                              callbacks=[check_point, early_stop, rop],
                              validation_steps=valid_generator.classes.size // BATCH_SIZE,
                              )

model.evaluate_generator(generator=valid_generator,steps=valid_generator.classes.size // BATCH_SIZE)



