# https://github.com/serengil/deepface
import os
from operator import itemgetter
from pathlib import Path
from deepface import DeepFace
BASE_DIR = Path(__file__).resolve().parent.parent


def DeepFace_recognize_organization(phone):

    user_test_image_dir = os.path.join(BASE_DIR,'media/user',phone,'test')
    all_user_phones = [dir for dir in os.listdir(os.path.join(BASE_DIR,'media/user')) if not dir.startswith('.')]

    distances = []
    for img in os.listdir(user_test_image_dir):
        test_img = user_test_image_dir+'/'+ img
    for i, phone in enumerate(all_user_phones):
        anchor_dir = os.path.join(BASE_DIR,'media/user',phone,'anchor')
        for img in os.listdir(anchor_dir):
            anchor_img = anchor_dir+'/'+ img
            try:
                result_dic = DeepFace.verify(
                    img1_path = test_img, img2_path = anchor_img,
                    detector_backend='ssd', model_name='Facenet512')
            except ValueError:
                return ('找不到脸部')
            distances.append((result_dic['distance'], phone))
    recog_distances, user_phone = min(distances, key=itemgetter(0))

    #remove the test photo
    os.remove(test_img)
    
    print(recog_distances)
    if recog_distances < 0.2:
        print('返回用户电话号码：{}'.format(user_phone))
        return(user_phone)
        
    else:
        return('无法识别身份')