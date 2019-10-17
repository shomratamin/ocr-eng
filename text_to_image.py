import sys
from qtpy.QtWidgets import QApplication, QTextEdit
from qtpy.QtGui import QPixmap, QFont, QFontMetrics, QPainter, QFontDatabase, QImage, QColor
from qtpy.QtCore import QRect, Qt, QPoint, qInstallMessageHandler
import numpy as np
import cv2
import copy
import time
from glob import glob

from random import shuffle, randint

import tensorflow as tf
from six import b

font_database_eng = []
backgrounds = []


def handle_qt_debug_message(level, context, message_bytes):
    pass


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def qt_pixmap_to_numpy_array(pixmap):
  
    image = pixmap.toImage()
    channels_count = image.format()
    height = image.height()
    width = image.width()
    s = image.bits().asstring(width * height * channels_count)
    arr = np.fromstring(s, dtype=np.uint8).reshape((height, width, channels_count))
    arr = cv2.cvtColor(arr,cv2.COLOR_RGBA2BGR)
    arr = np.array(arr,dtype=np.uint8())
    return arr

def create_tfrecord_data(generated_images, out_file):
    writer = tf.python_io.TFRecordWriter(out_file)

    for i, image in enumerate(generated_images,1):
        buff = cv2.imencode('.jpg', image[0])[1].tostring()
        label = image[1].encode('utf-8')
        feature = {}
        feature['image'] = _bytes_feature(buff)
        feature['label'] = _bytes_feature(label)
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

        if i % 1000 == 0:
            print('image {} encoded'.format(i))

    writer.close()

def initiate_fonts():
    global font_database_eng

    fonts = glob('fonts\\eng\\*.ttf')
    for _font in fonts:
        font_id = QFontDatabase.addApplicationFont(_font)
        family = QFontDatabase.applicationFontFamilies(font_id)
        monospace = QFont(family[0])
        monospace.setPixelSize(36)
        fm = QFontMetrics(monospace)
        font_database_eng.append([fm, monospace])

def initiate_backgrounds():
    global backgrounds
    files = glob('backgrounds\\*.jpg')
    for f in files:
        im = cv2.imread(f)
        backgrounds.append(im)


def add_random_noise(image):
    noise_typ = randint(0,6)

    if noise_typ == 0:
        row,col,ch= image.shape
        mean = 0
        var = 0.01
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == 1:
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == 2:
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    else:
        return image

    return noisy

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def HalfToneImage(imageGray):
    height, width = imageGray.shape
    imageHalfTone = np.zeros((2*height, 2*width)).astype(np.uint8)

    dict = {0:[[0,0],[0,0]],
            51:[[255,0],[0,0]],
            102:[[0,255],[255,0]],
            153:[[255,255],[255,0]],
            204:[[255,255],[255,255]]}

    for row in range(height):
        for col in range(width):
            val = imageGray[row][col]
            if(val > 204):
                imageHalfTone[row*2:row*2+2, col*2:col*2+2] = dict[204]
            elif(val >153):
                imageHalfTone[row*2:row*2+2, col*2:col*2+2] = dict[153]
            elif(val > 102):
                imageHalfTone[row*2:row*2+2, col*2:col*2+2] = dict[102]
            elif(val > 51):
                imageHalfTone[row*2:row*2+2, col*2:col*2+2] = dict[51]
            else:
                imageHalfTone[row*2:row*2+2, col*2:col*2+2] = dict[0]

    return imageHalfTone

def resize_randomly(image):
    h, w, c = image.shape
    height = randint(14,28)
    new_width = int((height * w)/h)
    if new_width < 5 or height < 5:
        return image
    image = cv2.resize(image,(new_width,height))
    padding = randint(0,8)
    top = randint(0,2)
    left = randint(0,2)
    bottom = randint(0,2)
    right = randint(0,2)
    if padding > 4:
        image = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(255,255,255))

    # do_thresh = randint(0,10)
    # if do_thresh == 1:
    #     image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #     image = cv2.threshold(image,128,255,cv2.THRESH_OTSU)[1]
    #     image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    # random_background = randint(3,3)
    # back_val = 170
    # if random_background == 3:
    #     back_val = randint(70,140)
    #     image[image > 220] = back_val
        

    # add_f_background = randint(0,7)
    # if add_f_background == 1:
    #     image = cv2.bitwise_not(image)
        # w,h,c = image.shape
        # background_id = randint(0, len(backgrounds)-1)
        # background_image = backgrounds[background_id]
        # wb,hb,cb = background_image.shape
        # _w = wb - w
        # _h = hb - h
        # if _w > 1 and _h > 1:
        #     _w = randint(0, _w)
        #     _h = randint(0, _h)
        #     background_image = background_image[_w:w+_w,_h:h+_h]
        #     image = cv2.bitwise_and(image,background_image)

    do_blur = randint(0,15)
    if do_blur == 0 and height > 13:
        image = cv2.GaussianBlur(image,(3,3),0)
    # elif do_blur == 1:
    #     image = cv2.medianBlur(image,3)
    # elif do_blur == 2:
    #     image = cv2.GaussianBlur(image,(3,3),0)
    #     image = cv2.medianBlur(image,3)

    # do_half_tone = randint(0,70)
    # if do_half_tone == 4:
    #     image = cv2.merge([HalfToneImage(x) for x in cv2.split(image)])

    # do_consecutive_morph = randint(0,20)
    # if do_consecutive_morph == 1:
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,2))
    #     image = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
    #     do_dilate = randint(0,2)
    #     if do_dilate == 1:
    #         image = cv2.morphologyEx(image,cv2.MORPH_DILATE,kernel)

    # image = add_random_noise(image)

    # do_morph = randint(0,8)
    # operations = [cv2.MORPH_DILATE, cv2.MORPH_ERODE, cv2.MORPH_CLOSE]
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,1))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(1,3))
    # kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    # kernels = [kernel1,kernel2, kernel3]
    # kernel_no = randint(0,2)
    # if do_morph < 2:
    #     image = cv2.morphologyEx(image,operations[do_morph],kernels[kernel_no])
    # if do_morph == 3:
    #     image = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel3)
    
    rotate_image = randint(0,11)
    if rotate_image == 5:
        angle = randint(-2,2)
        image = cv2.bitwise_not(image)
        image = rotateImage(image,angle)
        image = cv2.bitwise_not(image)

    # do_thresh_post = randint(0,3)
    # if do_thresh_post == 1 and back_val > 180:
    #     image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #     image = cv2.threshold(image,128,255,cv2.THRESH_OTSU)[1]
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    return image


def image_to_text(text_lines_eng):

    shuffle(text_lines_eng)
    text_lines_eng = text_lines_eng[:]
    text_lines = []
    text_lines.extend(text_lines_eng)
    generated_images = []
    counter = 0
    counter_randomize = 0
    max_length_of_line = 0
    max_char_in_line = 0
    max_char_line_no = 0
    image_length_limit_cross = 0
    font_database = [font_database_eng]
    shuffle(text_lines)
    font_db_id = 0
    for line in text_lines:
        counter += 1
        if counter == 4:
            break

        add_rand_space = randint(0,15)
        if add_rand_space == 8:
            _times = randint(1,3)
            space_counts = ['  ', '    ', '      ']
            _space_c = randint(0,2)
            line = line.replace(' ', space_counts[_space_c], _times)
        for x in range(10):
            current_font_id = randint(0,len(font_database[font_db_id]) - 1)
            fm, monospace = font_database[font_db_id][current_font_id][0], font_database[font_db_id][current_font_id][1]
            _set_bold = randint(0,3)
            if _set_bold == 2:
                monospace.setBold(True)
            line = line.replace('\n','')
            width=fm.width(line)
            height=fm.height()
            if width < 2 or height < 2:
                print('image is not generated at line', counter)
                width, height = randint(50, 350), 32
                line = ''
            pix = QPixmap(width,height)
            _back_color = 255
            pix.fill(QColor(_back_color,_back_color,_back_color))
            painter = QPainter(pix)
            _from = randint(0,50)
            _color = randint(_from,150)
            if _color >= _back_color:
                _color = 50
            painter.setBrush(QColor(_color,_color,_color))
            painter.setFont( monospace )
            _rect = QRect(0,0,width,height)
            painter.drawText(_rect, Qt.AlignVCenter | Qt.AlignHCenter, line)
            painter.end()
            np_arr = qt_pixmap_to_numpy_array(pix)
            if counter_randomize % 3 != 0:
                np_arr = resize_randomly(np_arr)
            h, w, c = np_arr.shape
            if h != 32:
                new_width = int((32 * w)/h) + 1
                if new_width > max_length_of_line:
                    max_length_of_line = new_width
                if new_width > 350 or len(line) > 34:
                    image_length_limit_cross += 1
                    continue
                np_arr = cv2.resize(np_arr,(new_width,32))
            # np_arr = cv2.cvtColor(np_arr, cv2.COLOR_BGR2GRAY)
            if len(line) > max_char_in_line:
                max_char_in_line = len(line)
                max_char_line_no = counter
            generated_images.append([np_arr,line])
            # print('counter {} font {}'.format(counter_randomize,monospace.family()))
            counter_randomize += 1


    for i, value in enumerate(generated_images):
        cv2.imwrite('./test_data/'+ str(i) + '.jpg', value[0])

    # shuffle(generated_images)
    # shuffle(generated_images)
    # data_split = int((len(generated_images) * .8))
    # dataset_ = read_dataset('datasets/dataset2',',')
    # generated_images.extend(dataset_)
    # dataset_ = read_dataset('datasets/dataset1')
    # generated_images.extend(dataset_)
    # dataset_ = read_dataset('datasets/dataset3')
    # generated_images.extend(dataset_)
    # shuffle(generated_images)
    # shuffle(generated_images)

    create_tfrecord_data(generated_images,'./train_data/training.tfrecords')
    # create_tfrecord_data(generated_images, './train_data/testing_all.tfrecords')


    print('train data count : {} \n'.format(len(generated_images)))
    print('max char in line {} at line no {}'.format(max_char_in_line, max_char_line_no))
    print('max image width {}'.format(max_length_of_line))
    print('over width data count: {}'.format(image_length_limit_cross))
    print('total {} images generated'.format(len(generated_images)))
    print('please wait a while to allow de-allocate memory.')


def images_to_tfrecords():
    file_object = open('edited/annot.txt', 'r', encoding='utf-8')
    lines = file_object.readlines()
    file_object.close()

    generated_data = []
    for line in lines:
        line = line.replace('\n', '')
        data = line.split(';')
        if len(data) == 2 and len(data[-1]) > 2:
            img_file = 'edited/' + str(data[0]) + '.jpg'
            image = cv2.imread(img_file)
            h,w, c = image.shape
            if w <= 350:
                generated_data.append([image,data[1]])
            if len(data[1]) > 36:
                print(line)

    create_tfrecord_data(generated_data,'./train_data/training.tfrecords')

def get_image_labels():
    file_object = open('edited/annot.txt', 'r', encoding='utf-8')
    lines = file_object.readlines()
    file_object.close()

    generated_data = []
    for line in lines:
        line = line.replace('\n', '')
        data = line.split(';')
        if len(data) == 2 and len(data[-1]) > 2:
            img_file = 'edited/' + str(data[0]) + '.jpg'
            image = cv2.imread(img_file)
            _line_text = data[1]
            _line_text = _line_text.strip()
            h,w, c = image.shape
            if w <= 350:
                generated_data.append([image,_line_text])
            if len(data[1]) > 36:
                print(line)

    return generated_data

def read_dataset(folder, separator=':', annot_file='annot.txt'):
    lines = []
    with open('{}/{}'.format(folder,annot_file), 'r', encoding='utf-8') as f:
        lines = f.readlines()

    generated_data = []
    for line in lines:
        line = line.replace('\n', '')
        data = line.split(separator)
        if len(data) >= 2:
            image_base_name = data[0]
            _line_text = separator.join(data[1:])
            img_file = '{}/{}.jpg'.format(folder,image_base_name)
            try:
                image = cv2.imread(img_file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                # print(img_file, _line_text)
                h,w = image.shape[:2]
                if w <= 350 and len(_line_text) <= 36 and len(_line_text) > 3:
                    generated_data.append([image,_line_text])
                    if len(_line_text) > 36:
                        print(line)
            except:
                continue

    return generated_data

def count_max_line(folder, separator=':', annot_file='annot.txt'):
    lines = []
    with open('{}/{}'.format(folder,annot_file), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    max_len = 0
    generated_data = []
    for line in lines:
        line = line.replace('\n', '')
        data = line.split(separator)
        if len(data) >= 2:
            image_base_name = data[0]
            _line_text = separator.join(data[1:])
            img_file = '{}/{}.jpg'.format(folder,image_base_name)
            try:
                image = cv2.imread(img_file)
                # print(img_file, _line_text)
                h,w = image.shape[:2]
                if w <= 350:
                    generated_data.append([image,_line_text])
                    if len(_line_text) > max_len:
                        max_len = len(_line_text)
            except:
                continue
    print('max line length',max_len)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    qInstallMessageHandler(handle_qt_debug_message)
    
    start = time.time()
    text_file_object_e = open('train_data/out_eng.txt','r', encoding='utf-8')
    text_lines_eng = text_file_object_e.readlines()

    shuffle(text_lines_eng)
    # fonts = glob('fonts\\ben\\*.ttf')
    # fonts.extend(glob('fonts\\ben\\*.ttf'))
    initiate_fonts()
    # initiate_backgrounds()
    image_to_text(text_lines_eng)

    end = time.time()
    total = end - start
    print('time taken : ', total, ' seconds')
    app.exit()
    # _ = count_max_line('datasets/dataset3')