import os
import cv2
import pytesseract
import PIL.Image

# Instal tesseract engine available in https://github.com/tesseract-ocr/tesseract

cwd = os.getcwd()


def del_dir_files(dir_path):
    for file in os.listdir(dir_path):
        os.remove(os.path.join(dir_path,file))


def gaussian_and_otsu(img_obj):
    blur = cv2.GaussianBlur(img_obj, (5,5), 0)
    ret3,th3 = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th3


def adaptive_threshold(img_obj):
    blur = cv2.GaussianBlur(img_obj, (5, 5), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 8)


def equalize_hist(img_obj):
    lab_img = cv2.cvtColor(img_obj, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    equa = cv2.equalizeHist(l)

    equalized_lab_img = cv2.merge((equa, a, b))
    equalized_img = cv2.cvtColor(equalized_lab_img, cv2.COLOR_LAB2BGR)

    return equalized_img


def CLAHE(img_obj):
    lab_img = cv2.cvtColor(img_obj, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab_img)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(l)

    clahe_lab_img = cv2.merge((clahe_img, a, b))
    clahe_img = cv2.cvtColor(clahe_lab_img, cv2.COLOR_LAB2BGR)

    return clahe_img


def get_gray_img(img_obj):
    return cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)


def get_video_frames(video_file_name, test=False):
    print(f'\n=> Generating frames from file \'{video_file_name}\'')
    video = cv2.VideoCapture(video_file_name)

    counter = 0
    video_frame_rate = 30
    img_number = 0

    while(True):
        ok, frame = video.read()

        if not ok:
            break

        if test and img_number >= 50:
            break

        #if counter == 0:
        if counter == video_frame_rate:
            equalized_img = equalize_hist(frame)
            #CLAHE_img = CLAHE(frame)
            gray_img = get_gray_img(equalized_img)
            #gaussian_ostu = gaussian_and_otsu(gray_img)
            gaussia_binary = adaptive_threshold(gray_img)

            img_path = 'frames/' + str(img_number) + '.jpg'

            cv2.imwrite(img_path, gaussia_binary)
            #print(f'saved: {img_path}')
            img_number += 1
            counter = 0

        counter += 1

    frames_path = os.path.join(cwd, 'frames')
    print(f'\n=> Frames generated at folder \'{frames_path}\'')
    video.release()
    return True


if __name__ == '__main__':
    frames_path = os.path.join(cwd, 'frames')
    if not os.path.isdir(frames_path):
        os.mkdir(frames_path)

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
        get_video_frames('video_2023-11-01_12-08-43.mp4', test=True)

    #tesseract 544.jpg stdout --psm 7 --oem 3 digits
    '''configs = '--psm 7 --oem 3 digits'

    list_frames = os.listdir(frames_path)
    numbers = []
    for frame in list_frames[:50]:
        file = 'frames/' + frame
        text = pytesseract.image_to_string(PIL.Image.open(file), config=configs)
        numbers.append(text)

    for ind in range(50):
        print(f'{ind} - \'{numbers[ind]}\'')'''