import os
import cv2
import pytesseract
import PIL.Image
import re

from image_handler import img_filters


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
    video_frame_rate = 30
    img_number = 0

    while(True):
        ok, frame = video.read()

        if not ok:
            break

        if test and img_number >= 50:
            break

        #if counter == 0:
        if counter == video_frame_rate: #verificar apenas um dado por segundo
            equalized_img = img_filters.equalize_hist(frame)
            #CLAHE_img = img_filters.CLAHE(frame)
            gray_img = img_filters.get_gray_img(equalized_img)
            #gaussia_binary = img_filters.adaptive_threshold(gray_img)

            img_path = 'frames/' + str(img_number) + '.jpg'
            cv2.imwrite(img_path, gray_img)
            #print(f'saved: {img_path}')
            img_number += 1
            counter = 0

        counter += 1

    print(f'\n=> Frames generated at folder \'{frames_path}\'')
    video.release()
    return True


def get_num_vals():
    num_vals = []
    for img in os.listdir(frames_path):
        img_path = os.path.join('frames', img)
        #img_path2 = os.path.join(frames_path, img) não funciona passar o full path pra imread por motivos de não sei
        img_obj = cv2.imread(img_path)
        #gray_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        gray_img = img_filters.get_gray_img(img_obj)
        #ret, th = img_filters.gaussian_and_otsu(gray_img)
        th = img_filters.adaptive_threshold(gray_img)

        cv2.imshow("aaa", th)
        cv2.waitKey(0)

        contours = img_filters.get_contours(th)
        #im2 = gray_img.copy()
        im2 = img_obj.copy()

        num = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            height, width, channels = im2.shape
            # if height of box is not tall enough relative to total height then skip
            #if (height / float(h)) > 6: continue

            ratio = h / float(w)
            # if height to width ratio is less than 1.5 skip
            if ratio < 1.5: continue

            # if width is not wide enough relative to total width then skip
            if width / float(w) > 15: continue

            #area = h * w
            # if area is less than 100 pixels skip
            #if area < 100: continue

            # draw the rectangle
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("aaa", rect)
            cv2.waitKey(0)

            # grab character region of image
            roi = th[y - 5:y + h + 5, x - 5:x + w + 5]
            # perfrom bitwise not to flip image to black text on white background
            roi = cv2.bitwise_not(roi)
            # perform another blur on character region
            roi = cv2.medianBlur(roi, 5)

            cv2.imshow("aaa", roi)
            cv2.waitKey(0)

            configs = '-c tessedit_char_whitelist=0123456789. --psm 7 --oem 3 digits'
            '''try:
                text = pytesseract.image_to_string(roi, config=configs)
                num_found = re.sub('[\d.]+', '', text) #Regular Expression to match only digits and dot
            except:
                num_found = None'''

            num.append(pytesseract.image_to_string(roi, config=configs))
            '''if num_found != None:
                num.append(num_found)'''

        num_vals.append(num)

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

        get_video_frames('video_2023-11-01_12-08-43.mp4', test=True)

    numbers = get_num_vals()
    '''for frame in list_frames:
        file = 'frames/' + frame
        text = pytesseract.image_to_string(PIL.Image.open(file), config=configs)
        numbers.append(text)'''

    for ind in range(50):
        print(f'{ind}.png => \'{numbers[ind]}\'')