#https://github.com/nicknochnack/FaceRecognition/blob/main/Facial%20Verification%20with%20a%20Siamese%20Network%20-%20Final.ipynb

import os
import shutil
from operator import itemgetter
import time
import uuid
from pathlib import Path
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall


cur_dir = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent
anchor_path = os.path.join(cur_dir,'data/anchor')
positive_path = os.path.join(cur_dir,'data/positive')
negative_path = os.path.join(cur_dir,'data/negative')

#collect anchor and positive images for training purpose
def collect_anchor_pos_images(anchor_path, positive_path):
    # Establish a connection to the webcam
    cap = cv2.VideoCapture(0)
    while cap.isOpened(): 
        ret, frame = cap.read()
    
        # Cut down frame to 250x250px
        frame = frame[200:200+400,500:500+400, :]
        
        # Collect anchors 
        if cv2.waitKey(1) & 0XFF == ord('a'):
            # Create the unique file path 
            imgname = os.path.join(anchor_path, '{}.jpg'.format(uuid.uuid1()))
            # Write out anchor iqmage
            cv2.imwrite(imgname, frame)
        
        # Collect positives
        if cv2.waitKey(1) & 0XFF == ord('p'):
            # Create the unique file path 
            imgname = os.path.join(positive_path, '{}.jpg'.format(uuid.uuid1()))
            # Write out positive image
            cv2.imwrite(imgname, frame)
        
        # Show image back to screen
        cv2.imshow('Image Collection', frame)
        
        # Breaking gracefully
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
            
    # Release the webcam
    cap.release()
    # Close the image show frame
    cv2.destroyAllWindows()


#data augmentation -- augmenting the collected data to generate more training data
def image_aug(img):
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
        # img = tf.image.stateless_random_crop(img, size=(20,20,3), seed=(1,2))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100)))
            
        data.append(img)
    
    return data

def augment_images(image_paths):
    for img_path in image_paths:
        print('augmenting image {}...'.format(img_path))
        img = cv2.imread(img_path)
        augmented_images = image_aug(img) 
        
        for image in augmented_images:
            image_name = '{}.jpg'.format(uuid.uuid1())
            new_path = '/'.join(img_path.split('/')[:-1])
            print('...save augmented image {}'.format(os.path.join(new_path,image_name)))
            cv2.imwrite(os.path.join(new_path, image_name), image.numpy())



def image_preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)

    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))

    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img


def image_preprocess_twin(input_img,validation_img,label):
    return(image_preprocess(input_img),image_preprocess(validation_img),label)


#create embedding layer
def make_embedding(): 
    inp = Input(shape=(100,100,3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')


# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


#create siamese model
def make_siamese_model(): 

    embedding = make_embedding()
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


#train step function
@tf.function
def train_step(batch, opt, loss_func, siamese_model):
    
    # Record all of our operations 
    with tf.GradientTape() as tape:     
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = loss_func(y, yhat)
    print(loss)
        
    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
        
    # Return loss
    return loss


#training loops
def train(data, EPOCHS, siamese_model, learning_rate=0.0001):

    # setup loss and optimizer
    print('setup loss and optimizer')
    binary_cross_loss = tf.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam(learning_rate)  

    # establish checkpoints
    print('create checkpoints...')
    checkpoint_dir = os.path.join(cur_dir,'checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt')
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model) 

    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # Creating a metric object 
        r = Recall()
        p = Precision()
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            loss = train_step(batch, opt, binary_cross_loss, siamese_model)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
        
        # Save checkpoints
        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)


def train_initial_model(anchor_path, positive_path, negative_path, learning_rate=0.0001):

    #create the training datasets
    print('create the training datasets...')
    anchor = tf.data.Dataset.list_files(anchor_path+'/*.jpg').take(300)
    positive = tf.data.Dataset.list_files(positive_path+'/*.jpg').take(300)
    negative = tf.data.Dataset.list_files(negative_path+'/*.jpg').take(300)

    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    data = positives.concatenate(negatives)

    # build train and test partition
    print('build the train and test data partitions...')
    data = data.map(image_preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=10000)

    # training partition
    train_data = data.take(round(len(data)*0.7))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    # testing partition
    test_data = data.skip(round(len(data)*0.7))
    test_data = test_data.take(round(len(data)*0.3))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)

    ## training
    # Siamese model
    print('make the siamese model...')
    siamese_model = make_siamese_model()

    # train the model

    print('training the model...')
    time.sleep(5)
    EPOCHS = 50
    train(train_data, EPOCHS, siamese_model, learning_rate)

    #make predictions
    r = Recall()
    p = Precision()

    for test_input, test_val, y_true in test_data.as_numpy_iterator():
        yhat = siamese_model.predict([test_input, test_val])
        r.update_state(y_true, yhat)
        p.update_state(y_true,yhat) 

    print('The testing recall {}...precision {}'.format(r.result().numpy(), p.result().numpy()))
    time.sleep(10)

    # Save weights
    save_path = os.path.join(cur_dir,'siamesemodelv2.h5')
    print('save the newly trained network to {}'.format(save_path))
    siamese_model.save(save_path)


# upon receiving the posted face images, 
# make a copy and put them in /media/user/phone/anchor before augmenting them
def copy_face_images(user_phone):

    uploaded_photo_dir = os.path.join(BASE_DIR,'media/user',user_phone)
    images = [img for img in os.listdir(uploaded_photo_dir) if img.endswith('jpg')]
    user_anchor_path = os.path.join(BASE_DIR,'media/user',user_phone,'anchor')
    if not os.path.exists(user_anchor_path):
        os.makedirs(user_anchor_path)
    else:
        existing_images = os.listdir(user_anchor_path)
        for img in existing_images:
            os.remove(os.path.join(user_anchor_path,img))
    for img in images:
        img_path = os.path.join(uploaded_photo_dir,img)
        new_path = os.path.join(user_anchor_path,img)
        shutil.copy(img_path,new_path)


#update the model for a newly registered user
def update_model_with_new_training_data(user_phone, learning_rate=0.00001):

    #get all the newly updated images' path
    uploaded_photo_dir = os.path.join(BASE_DIR,'media/user',user_phone)
    print('get the configured images paths at {}'.format(uploaded_photo_dir))
    time.sleep(1)
    images = (img for img in os.listdir(uploaded_photo_dir) if img.endswith('jpg'))
    image_paths = (os.path.join(uploaded_photo_dir,img) for img in images)

    #augment the uploaed images
    print('augmenting the images...')
    augment_images(image_paths)

    num_images = len([img for img in os.listdir(uploaded_photo_dir) if img.endswith('jpg')]) // 2
    augmented_images = tf.data.Dataset.list_files(uploaded_photo_dir+'/*.jpg')
    anchor = augmented_images.take(num_images)
    positive = augmented_images.take(num_images)
    negative = tf.data.Dataset.list_files(negative_path+'/*.jpg').take(num_images)

    #create the training data
    print('create the learning data...')
    time.sleep(1)
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor))) ))
    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    
    data = positives.concatenate(negatives)

    data = data.map(image_preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=10000)

    #training partition
    train_data = data.take(round(len(data)*0.7))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    # testing partition
    test_data = data.skip(round(len(data)*0.7))
    test_data = test_data.take(round(len(data)*0.3))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)

    #reload model
    model_path = os.path.join(cur_dir,'siamesemodelv2.h5')
    print('reload the existing model at {}...'.format(model_path))
    siamese_model = tf.keras.models.load_model(model_path, 
                    custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
    time.sleep(1)

    # train the model
    EPOCHS = 3
    train(train_data, EPOCHS, siamese_model, learning_rate)

    #make predictions
    r = Recall()
    p = Precision()

    for test_input, test_val, y_true in test_data.as_numpy_iterator():
        yhat = siamese_model.predict([test_input, test_val])
        r.update_state(y_true, yhat)
        p.update_state(y_true, yhat) 

    print('The testing recall {}...precision {}'.format(r.result().numpy(), p.result().numpy()))
    time.sleep(1)

    # Save weights
    save_path = os.path.join(cur_dir,'siamesemodelv2.h5')
    print('save the newly trained network to {}'.format(save_path))
    siamese_model.save(save_path)

    #remove all configuration images
    images = (img for img in os.listdir(uploaded_photo_dir) if img.endswith('jpg'))
    image_paths = (os.path.join(uploaded_photo_dir,img) for img in images)
    for img_path in image_paths:
        os.remove(img_path)



#recognize one image   
def recognize(test_img, anchor_img):

    #reload model
    model_path = os.path.join(cur_dir,'siamesemodelv2.h5')
    siamese_model = tf.keras.models.load_model(model_path, 
                    custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

    y_pred = siamese_model.predict([test_img, anchor_img])

    return(y_pred)

    
#pass through all anchor images, pick the most similar one,
#compare it with the 0.8 threshold
#if >= 0.8, then return the corresponding phone_number of the user
#if < 0.8, then return 'unrecognized'
def recognize_organization(phone):

    user_test_image_dir = os.path.join(BASE_DIR,'media/user',phone,'test')
    all_user_phones = [dir for dir in os.listdir(os.path.join(BASE_DIR,'media/user')) if not dir.startswith('.')]

    probabilities = []
    test_img = tf.data.Dataset.list_files(user_test_image_dir+'/*.jpg').take(1)
    for i, phone in enumerate(all_user_phones):
        anchor_dir = os.path.join(BASE_DIR,'media/user',phone,'anchor')
        anchor_images = tf.data.Dataset.list_files(anchor_dir+'/*.jpg')
        for j in range(3):
            anchor_img = anchor_images.take(1)
            data = tf.data.Dataset.zip((test_img, anchor_img, tf.data.Dataset.from_tensor_slices(tf.zeros(1))))
            data = data.map(image_preprocess_twin)
            data = data.batch(1)
            for test_input, anchor_input, label in data.as_numpy_iterator():
                pred_prob = recognize(test_input, anchor_input)
                probabilities.append((pred_prob[0][0],phone))

    recog_prob, user_phone = max(probabilities, key=itemgetter(0))

    #remove the test photo
    for img in os.listdir(user_test_image_dir):
        os.remove(os.path.join(user_test_image_dir,img))
    
    if recog_prob > 0.6:
        print('返回用户电话号码：{}'.format(user_phone))
        return(user_phone)
        
    else:
        return('unrecognized identity!')



if __name__ == '__main__':

    print('debugging...all functions')

    ## collect 
    #collect anchor images and positive images
    # collect_anchor_pos_images(anchor_path,positive_path)

    #generate augmented images for training purpose
    # positive_image_paths = [os.path.join(positive_path,img) for img in os.listdir(positive_path) if not img.startswith('.')]
    # anchor_image_paths = [os.path.join(anchor_path,img) for img in os.listdir(anchor_path) if not img.startswith('.')]
    # augment_images(positive_image_paths)
    # augment_images(anchor_image_paths)
    # print(len(positive_image_paths))
    # print(len(anchor_image_paths))
    # print(len(os.listdir(os.path.join(negative_path))))

    #train initial model
    # train_initial_model(anchor_path, positive_path, negative_path, learning_rate=0.0001)

    #test copy images from media/user/phone to media/user/phone/anchor
    # copy_face_images('123')

    #test 
    # update_model_with_new_training_data('123', learning_rate=0.00001)

    #test predictiton
    # recognize_organization('123')
