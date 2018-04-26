from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

BATCH_SIZE = 16

model = load_model('model.hdf5')

test_data_gen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_data_gen.flow_from_directory(directory='assets/',
                                                     target_size=(224, 224),
                                                   classes=['test'],
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='categorical',shuffle=False)

prediction = model.predict_generator(test_generator)

results = {}
results['fn'] = test_generator.filenames
results['id_card'] = prediction[:,0]
results['no_id_card'] = prediction[:,1]

df = pd.DataFrame(results)
df = df.set_index('fn')
df.to_csv('prediction.csv')