
import pandas as pd
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.applications import *
from keras.preprocessing.image import *
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.applications.inception_v3 import preprocess_input
import os
import random
import imghdr
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
classes = ['collar_design_labels', 'skirt_length_labels','lapel_design_labels',
           'sleeve_length_labels', 'neckline_design_labels','neck_design_labels', 'coat_length_labels',
           'pant_length_labels']
for i in range(2):
    cur_class =classes[i+3]
    df_train = pd.read_csv('./train_2.csv', header=None)
    df_train.columns = ['image_id', 'class', 'label']
    df_train.head()
    df_load = df_train[(df_train['class'] == cur_class)].copy()
    df_load.reset_index(inplace=True)
    del df_load['index']
    print('{0}: {1}'.format(cur_class, len(df_load)))
    num_sample = len(df_load)
    num_class = len(df_load['label'][0])
    width = 299
    X = np.zeros((num_sample, width, width, 3), dtype=np.uint8)
    y = np.zeros((num_sample, num_class), dtype=np.uint8)

    for i in tqdm(range(num_sample)):
        tmp_label = df_load['label'][i]
        path = './train/{0}'.format(df_load['image_id'][i])
        if os.path.exists(path) == False:
            continue
        if len(tmp_label) > num_class:
            print(df_load['image_id'][i])
        path='./train/{0}'.format(df_load['image_id'][i])
        if os.path.exists(path)==False:
            continue
        if imghdr.what(path)==False:
            continue
        b=random.randint(0,1)
        if b==0:
            a = cv2.imread('./train/{0}'.format(df_load['image_id'][i]))
            a = cv2.resize(a, (width, width))
            X[i]=cv2.flip(a,1)
            y[i][tmp_label.find('y')] = 1
        else:
            a = cv2.imread('./train/{0}'.format(df_load['image_id'][i]))
            X[i] = cv2.resize(a, (width, width))
            y[i][tmp_label.find('y')] = 1

    #X_train, X_valid, y_train, y_valid = train_test_split(X, y,test_size=0.1,random_state=random.randint(1,100))
    n_valid=int(num_sample*0.1)
    X_train=X[n_valid:]
    X_valid=X[:n_valid]
    y_train=y[n_valid:,:]
    y_valid=y[:n_valid,:]
    index = [i for i in range(len(X_train))]
    np.random.shuffle(index)
    X_train = X_train[index]
    y_train = y_train[index]


    print('X_valid total num:'+str(len(X_valid)))
    print('y_valid total num:'+str(len(y_valid)))

    index = [i for i in range(len(X_valid))]
    np.random.shuffle(index)
    X_valid = X_valid[index]
    y_valid = y_valid[index]
    class Generator():
        def __init__(self, X, y, batch_size=32, aug=False):
            def generator():
                idg = ImageDataGenerator(horizontal_flip=True,
                                              rotation_range=20,
                                              width_shift_range=0.2,
                                              height_shift_range=0.2,
                                              fill_mode='nearest',
                                              zoom_range=0.2)
                while True:
                    for i in range(0, len(X), batch_size):
                        X_batch = X[i:i+batch_size].copy()
                        y_barch = y[i:i+batch_size,:].copy()
                        if aug:
                            for j in range(len(X_batch)):
                                X_batch[j] = idg.random_transform(X_batch[j])
                        yield X_batch, y_barch
            self.generator = generator()
            self.steps = len(X) // batch_size + 1

    gen_train = Generator(X_train, y_train, batch_size=16, aug=True)
    com_model =DenseNet201(include_top=False, input_shape=(width, width, 3), weights='imagenet')
    inputs = Input((width, width, 3))
    featuremap = inputs
    featuremap = Lambda(preprocess_input, name='preprocessing')(featuremap)
    featuremap = com_model(featuremap)
    featuremap = GlobalAveragePooling2D()(featuremap)
    featuremap = Dropout(0.5)(featuremap)
    prediction = Dense(num_class, activation='softmax',name='softmax')(featuremap)
    model = Model(inputs, prediction)

    prefix_cls = cur_class.split('_')[0]
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.000001, mode='auto', verbose=1)
    model.compile(optimizer=adam(1e-4),loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=80,callbacks=[EarlyStopping(patience=3)],validation_data=(X_valid, y_valid))
    model.compile(optimizer=adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps,epochs=80,callbacks=[EarlyStopping(patience=3)],validation_data=(X_valid, y_valid))
    model.compile(optimizer=adam(1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=80,callbacks=[EarlyStopping(patience=3)],validation_data=(X_valid, y_valid))
    model.save('./models/{}_Densnet201_jingxiang_gen_a_a_a_5.5.h5'.format(prefix_cls))
    model.evaluate(X_train, y_train, batch_size=32)
    model.evaluate(X_valid, y_valid, batch_size=32)
    df_test = pd.read_csv('test.csv', header=None)
    df_test.columns = ['image_id', 'class', 'x']
    del df_test['x']
    df_test.head()
    df_load = df_test[(df_test['class'] == cur_class)].copy()
    df_load.reset_index(inplace=True)
    del df_load['index']
    print('{0}: {1}'.format(cur_class, len(df_load)))
    df_load.head()
    n = len(df_load)
    X_test = np.zeros((n, width, width, 3), dtype=np.uint8)
    for i in tqdm(range(n)):
        X_test[i] = cv2.resize(cv2.imread('./test/{0}'.format(df_load['image_id'][i])), (width, width))
    test_np = model.predict(X_test, batch_size=32)
    print(test_np.shape)
    result = []
    for i, row in df_load.iterrows():
        tmp_list = test_np[i]
        tmp_result = ''
        for tmp_ret in tmp_list:
            tmp_result += '{:.4f};'.format(tmp_ret)
        result.append(tmp_result[:-1])
    df_load['result'] = result
    df_load.head()
    df_load.to_csv('./result/{}_Densnet201_jingxiang_gen_a_a_a_5.5.csv'.format(prefix_cls), header=None, index=False)

