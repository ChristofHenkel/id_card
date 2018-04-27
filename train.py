from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Input,Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

BATCH_SIZE = 16
COLOR = 'rgb' #greyscale


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
                                                     class_mode='categorical',color_mode = COLOR)

valid_data_gen = ImageDataGenerator(rescale=1. / 255)
valid_generator = valid_data_gen.flow_from_directory(directory='assets/valid/',
                                                     target_size=(224, 224),
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='categorical',color_mode = COLOR)


def set_model(lr, color = 'rgb',print_architecture=False):

    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    if color == 'greyscale':
        inp = Input(shape=(224, 224, 1))
        x = Concatenate()([inp, inp, inp])
        x = base_model(x)
        main = GlobalAveragePooling2D()(x)
        out = Dense(2, activation='softmax')(main)
        model = Model(inputs=inp, outputs=out)

    else:

        main = GlobalAveragePooling2D()(base_model.output)
        out = Dense(2, activation='softmax')(main)
        model = Model(inputs=base_model.input, outputs=out)

    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy',metrics=['accuracy'])
    if print_architecture:
        model.summary()
    return model


model = set_model(lr=0.0001, color=COLOR)

check_point = ModelCheckpoint('model.hdf5', monitor="val_loss", mode="min", save_best_only=True, verbose=1)
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



