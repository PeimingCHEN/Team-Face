import tensorflow as tf
from PIL import Image
from fr_algorithms.facenet.facenet import Facenet
from operator import itemgetter
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def FaceNet_recognize_organization(phone):
    model = Facenet()
    user_test_image_dir = os.path.join(BASE_DIR,'media/user',phone,'test')
    all_user_phones = [dir for dir in os.listdir(os.path.join(BASE_DIR,'media/user')) if not dir.startswith('.')]

    probabilities = []
    for img in os.listdir(user_test_image_dir):
        test_img_dir = user_test_image_dir+'/'+ img
        test_img = Image.open(test_img_dir)
    for i, phone in enumerate(all_user_phones):
        anchor_dir = os.path.join(BASE_DIR,'media/user',phone,'anchor')
        for img in os.listdir(anchor_dir):
            anchor_img = anchor_dir+'/'+ img
            anchor_img = Image.open(anchor_img)
            probability = model.detect_image(test_img, anchor_img)
            probabilities.append((float(probability), phone))
    recog_prob, user_phone = min(probabilities, key=itemgetter(0))

    #remove the test photo
    os.remove(test_img_dir)
    
    print(recog_prob)
    if recog_prob < 0.7:
        print('返回用户电话号码：{}'.format(user_phone))
        return(user_phone)
        
    else:
        return('无法识别身份')