from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
BATCH_SIZE = 16

train_data_gen = ImageDataGenerator(rescale=1./255,
                                    vertical_flip=True,
                                    horizontal_flip=True,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    zoom_range=0.2,
                                    shear_range=0.2,
                                    )

train_generator = train_data_gen.flow_from_directory(directory='assets/data/',
                             target_size=(224,224),
                            batch_size=BATCH_SIZE,
                             class_mode='categorical')

#valid_data_gen = ImageDataGenerator(rescale=1./255)
#valid_generator = valid_data_gen.flow_from_directory(directory='assets/valid/',
#                             target_size=(224,224),
#                            batch_size=BATCH_SIZE,
#                             class_mode='categorical')#

base_model = VGG16(weights='imagenet', include_top=False,input_shape=(224,224,3))
main = GlobalAveragePooling2D()(base_model.output)
out = Dense(2, activation='softmax')(main)







model = Model(inputs=base_model.input, outputs = out)
model.compile(optimizer=Adam(lr = 0.0001), loss='categorical_crossentropy')
model.summary()


check_point = ModelCheckpoint('model.hdf5', monitor="val_loss", mode="min", save_best_only=True, verbose=1)
#early_stop = EarlyStopping(patience=3)
history = model.fit_generator(train_generator,
steps_per_epoch=train_generator.classes.size//BATCH_SIZE,
                              epochs=30,
                    #validation_data = data_gen_valid(),
                    callbacks=[check_point],
                    #validation_steps= 540,
                    )