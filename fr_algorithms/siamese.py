#https://github.com/nicknochnack/FaceRecognition/blob/main/Facial%20Verification%20with%20a%20Siamese%20Network%20-%20Final.ipynb

import os
from turtle import up
import uuid
from pathlib import Path
from backend.settings import BASE_DIR
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall


cur_dir = Path(__file__).resolve().parent
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
            print('...save augmented image {}'.format(image_name))
            cv2.imwrite(os.path.join(img_path, image_name), image.numpy())



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

    # establish checkpoints
    checkpoint_dir = os.path.join(cur_dir,'checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt')
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model) 

    # setup loss and optimizer
    binary_cross_loss = tf.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam(learning_rate)  

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
    anchor = tf.data.Dataset.list_files(anchor_path+'/*.jpg').take(30)
    positive = tf.data.Dataset.list_files(positive_path+'/*.jpg').take(30)
    negative = tf.data.Dataset.list_files(negative_path+'/*.jpg').take(30)

    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    data = positives.concatenate(negatives)

    # build train and test partition
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
    siamese_model = make_siamese_model()

    # train the model
    EPOCHS = 50
    train(train_data, EPOCHS, siamese_mmodel, learning_rate)

    #make predictions
    r = Recall()
    p = Precision()

    for test_input, test_val, y_true in test_data.as_numpy_iterator():
        yhat = siamese_model.predict([test_input, test_val])
        r.update_state(y_true, yhat)
        p.update_state(y_true,yhat) 

    print(r.result().numpy(), p.result().numpy())

    # Save weights
    siamese_model.save(os.path.join(cur_dir,'siamesemodelv2.h5'))



#update the model for a newly registered user
def update_model_with_new_training_data(user_phone, learning_rate=0.00001):

    #get all the newly updated images' path
    uploaded_photo_dir = os.path.join(BASE_DIR,'media/user',user_phone)
    images = [img for img in os.listdir(uploaded_photo_dir) if not img.startswith('.')]
    image_paths = [os.path.join(uploaded_photo_dir,img) for img in images]

    #augment the uploaed images
    augment_images(image_paths)
    augmented_images = [img for img in os.listdir(uploaded_photo_dir) if not img.startswith('.')]
    augmented_image_paths = [os.path.join(uploaded_photo_dir,img) for img in augmented_images]
    num_images = len(augmented_images)
    neg_image_paths = os.listdir(negative_path) #get the negative images' paths

    #split the images into anchor and positive sets, respectively
    num_anchor = num_images // 2

    anchor = augmented_image_paths[:num_anchor]
    positive = augmented_image_paths[num_anchor:(num_anchor*2)]
    negative = random.sample(neg_image_paths, num_anchor)

    #create the training data
    negatives = tf.data.Dataset.zip(anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor))))
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
    siamese_model = tf.keras.models.load_model(model_path, 
                    custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

    # train the model
    EPOCHS = 50
    train(train_data, EPOCHS, siamese_model, learning_rate)

    #make predictions
    r = Recall()
    p = Precision()

    for test_input, test_val, y_true in test_data.as_numpy_iterator():
        yhat = siamese_model.predict([test_input, test_val])
        r.update_state(y_true, yhat)
        p.update_state(y_true, yhat) 

    print(r.result().numpy(), p.result().numpy())

    # Save weights
    siamese_model.save(os.path.join(cur_dir,'siamesemodelv2.h5'))

    
def 


    


if __name__ == '__main__':

    ## collect 
    #collect anchor images and positive images
    collect_anchor_pos_images(anchor_path,positive_path)

    #generate augmented images for training purpose
    # positive_image_paths = [os.path.join(positive_path,img) for img in os.listdir(positive_path) if not img.startswith('.')]
    # anchor_image_paths = [os.path.join(anchor_path,img) for img in os.listdir(anchor_path) if not img.startswith('.')]
    # augment_images(positive_image_paths)
    # augment_images(anchor_image_paths)



