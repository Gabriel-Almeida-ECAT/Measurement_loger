import os
import cv2
import pytesseract
import re

from MeasumentLogerLib import img_filters


cwd = os.getcwd()
frames_path = os.path.join(cwd, 'frames')
if not os.path.isdir(frames_path):
    os.mkdir(frames_path)


def del_dir_files(dir_path):
    for file in os.listdir(dir_path):
        os.remove(os.path.join(dir_path,file))


def get_video_frames(video_path, test=False):
    print(f'\n=> Generating frames from file \'{video_path}\'')
    video = cv2.VideoCapture(video_path)

    counter = 0

    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(f'Frames per second: {fps}')

    #sample_rate = get_sample_rate() #take percentage as input, maximum is the video fps
    sample_rate = 1 #for debug porposes
    print(f'Amount of registers per second: {sample_rate}')
    img_number = 0

    while(True):
        ok, frame = video.read()

        if not ok:
            break

        if test and img_number >= 50:
            break

        #if counter == 0:
        if counter == fps/sample_rate: #verificar x dados por segundo
            equalized_img = img_filters.equalize_hist(frame)
            gray_img = img_filters.get_gray_img(equalized_img)

            median = img_filters.median_blur(gray_img)
            gaussia_binary = img_filters.adaptive_gaussian_threshold(median)
            inverted_img = img_filters.invert_img(gaussia_binary)
            #otsu = img_filters.gaussian_and_otsu(gray_img)

            #new_size = (28, 28)
            #resized_image = cv2.resize(inverted_img, new_size)

            img_path = 'frames/' + str(img_number) + '.jpg'
            cv2.imwrite(img_path, inverted_img)

            img_number += 1
            counter = 0

        counter += 1

    print(f'\n=> {img_number} frames generated at folder \'{frames_path}\'')
    video.release()
    return True


def get_num_vals():
    num_vals = []
    list_files = [os.path.join('frames', file) #full path doesnt work
                  for file in os.listdir(frames_path)
                  if os.path.isfile(os.path.join('frames', file))]

    for img in list_files:
        img_obj = cv2.imread(img)

        configs = '-c tessedit_char_whitelist=0123456789. --psm 7 --oem 3 digits'

        #text = pytesseract.image_to_string(PIL.Image.open(file), config=configs)
        text = pytesseract.image_to_string(img_obj, config=configs)
        #num_found = re.sub('[\d.]+', '', text)  # Regular Expression to match only digits and dot character => didin't work great

        num_vals.append(text)

    return num_vals


def get_num_vals_contours():
    num_vals = []
    for img in os.listdir(frames_path):
        img_path = os.path.join('frames', img) #full path doesnt work
        img_obj = cv2.imread(img_path)

        height, width, _ = img_obj.shape

        configs = '-c tessedit_char_whitelist=0123456789. --psm 6 --oem 3 digits'

        boxes = pytesseract.image_to_boxes(img_obj, config=configs)
        text = ''
        for box in boxes.splitlines():
            box = box.split(" ")
            img = cv2.rectangle(img_obj, (int(box[1]), height - int(box[2])), (int(box[3]), height - int(box[4])), (0, 0, 255))
            text += box[0]
        #text = pytesseract.image_to_string(img_obj, config=configs)

        cv2.imshow("img", img_obj)
        cv2.waitKey(0)

        num_vals.append(text)

    return num_vals


if __name__ == '__main__':
    generate_frames = False
    have_frames = True if len(os.listdir('frames')) > 0 else False
    if have_frames:
        while True:
            erase_ans = input('Frames found in folder, wish to generate new frames? [y/n]: ')
            erase_ans = erase_ans.upper()
            if erase_ans == 'Y' or erase_ans == 'N':
                break
    else:
        generate_frames = True
        erase_ans = 'N'

    if erase_ans == 'Y' and have_frames:
       del_dir_files(frames_path)
       generate_frames = True
    elif erase_ans == 'N' and have_frames:
        print('\n=> Executing script with current frames')

    if generate_frames:
        '''video_path = input('\nEnter video path: ')
        get_video_frames(video_path, test=True)'''

        get_video_frames('video_2023-11-01_12-08-43.mp4', test=False)

    '''numbers = get_num_vals()

    for ind in range(50):
        print(f'{ind}.png => \'{numbers[ind]}\'')'''