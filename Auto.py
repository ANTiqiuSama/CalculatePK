import inspect
import os
from PIL import Image, ImageOps
import pyautogui
import pygetwindow as gw
import cv2
import numpy as np
import time

script_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
WINDOW_TITLE = "Monkey CalculatingPK"
QUESTION_COUNT = 10
BUTTON_WIDTH = 300
BUTTON_HEIGHT = 100
ANSWER_WIDTH = 780
ANSWER_HEIGHT = 610
MAXIMUM_DIGIT = 4
DIGIT_MARGIN = 40

WINDOW_POSITION_OFFSET_X = 10
WINDOW_POSITION_OFFSET_Y = 45
QUESTION_REGION_OFFSET_X = 40
QUESTION_REGION_OFFSET_Y = 270
QUESTION_REGION_WIDTH = 770
QUESTION_REGION_HEIGHT = 80

WINDOW_WIDTH = 850
WINDOW_HEIGHT = 1400

WAIT_DURATION = 0.6

def draw_number(num, x, y):
    digits = []
    for digit in str(num):
        digits.append(int(digit))
    
    w = ANSWER_WIDTH // MAXIMUM_DIGIT - DIGIT_MARGIN
    h = ANSWER_HEIGHT // 3 * 2
    y = y + ANSWER_HEIGHT // 6 + DIGIT_MARGIN // 2
    h = h - DIGIT_MARGIN
    
    for i in range(len(digits)):
        x = x + DIGIT_MARGIN // 2
        segments = [
            [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)],                               # 0
            [(x + w // 2, y), (x + w // 2, y + h)],                                                 # 1
            [(x, y), (x + w, y), (x, y + h), (x + w, y + h)],                                       # 2
            [(x, y), (x + w, y + h // 4), (x, y + h // 2),(x + w, y + h // 4 * 3),(x, y + h)],      # 3
            [(x + w, y + h), (x + w, y), (x, y + h // 2), (x + w, y + h // 2)],                     # 4
            [(x + w, y), (x, y), (x, y + h // 2), (x + w, y + h // 2), (x + w, y + h), (x, y + h)], # 5
            [(x + w, y), (x, y + h // 2), (x + w, y + h), (x + w, y + h // 2), (x, y + h // 2)],    # 6
            [(x, y), (x + w, y), (x + w, y + h)],                                                   # 7
            [(x, y), (x + w, y), (x, y + h), (x + w, y + h), (x, y)],                               # 8
            [(x + w, y + h), (x + w, y), (x, y), (x, y + h // 2), (x + w, y + h // 4)],             # 9
        ]
        
        pyautogui.moveTo(segments[digits[i]][0][0], segments[digits[i]][0][1])
        pyautogui.mouseDown(button='left')
        for segment in segments[digits[i]]:
            pyautogui.moveTo(segment[0], segment[1])
        pyautogui.mouseUp(button='left')
        x = x + w + DIGIT_MARGIN // 2

def split_image(image):
    col_sum = np.sum(image, axis=0)
    col_has_pixel = np.where(col_sum > 255)[0]
    split_index = []
    result = []
    for col in col_has_pixel:
        if col_sum[col - 1] == 0 or (col != col_sum.shape[0] - 1 and col_sum[col + 1] == 0):
            split_index.append(col)
    print(f"len(split_index): {len(split_index)}")
    i = 0
    while i < len(split_index) - 1:
        print(f"split_index[{i}]: {split_index[i]}, {split_index[i + 1]}")
        result.append(image[:, split_index[i]:split_index[i + 1]])
        i += 2
    return result

def image_to_str(img,data):
    result = ""
    for src_img in split_image(img):
        for label in data:
            if src_img.shape == data[label].shape and ((src_img - data[label]) ** 2).mean() == 0:
                result += label
                break
    return result

def main():
    # Generate haracters data:
    # "0 1 2 3 4 5 6 7 8 9"
    # "+ - × ÷ ( ) ? ."
    data_img = []
    char_label = []
    char_data = {}
    data_img.append(np.array(ImageOps.invert(Image.open(script_directory + "\\res\\data_0.png")).convert("L")))
    data_img.append(np.array(ImageOps.invert(Image.open(script_directory + "\\res\\data_1.png")).convert("L")))
    char_label.append(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    char_label.append(["+", "-", "*", "/", "(", ")", "=", "?", "."])
    "+ - × ÷ ( ) = ? ."
    for i in range(2):
        idx = 0
        for img in split_image(data_img[i]):
            char_data[char_label[i][idx]] = img
            idx += 1
    
    target_window = None
    while target_window == None:
        for window in gw.getAllWindows():
            if WINDOW_TITLE in window.title:
                target_window = window
                break
    
    time.sleep(1)  # waiting to activate
    
    window_x = target_window.left + WINDOW_POSITION_OFFSET_X
    window_y = target_window.top + WINDOW_POSITION_OFFSET_Y
    
    width, height = target_window.width, target_window.height
    
    start_button_x = window_x + width // 2
    start_button_y = window_y + height * 0.7
    
    canvas_x = window_x + 35
    canvas_y = window_y + 635
    
    restart_button_x = window_x + width // 2
    restart_button_y = window_y + height * 0.8
    
    # window_region = (window_x, window_y, WINDOW_WIDTH, WINDOW_HEIGHT)
    question_region = (window_x + QUESTION_REGION_OFFSET_X, window_y + QUESTION_REGION_OFFSET_Y, QUESTION_REGION_WIDTH, QUESTION_REGION_HEIGHT)
    
    pyautogui.moveTo(start_button_x, start_button_y)
    time.sleep(0.5)
    pyautogui.click()
    
    for i in range(QUESTION_COUNT):
        # question_img = pyautogui.screenshot(region=question_region).convert('L').point(lambda x: 0 if x < 128 else 255) # save original image to make data
        question_img = ImageOps.invert(pyautogui.screenshot(region=question_region).convert('L')).point(lambda x: 0 if x < 128 else 255)
        question_img.save(f"question_{i}.png")
        question_text = image_to_str(np.array(question_img), char_data)
        question_text = question_text.split("=")[0].replace("×","*").replace("÷","/")
        print(f"QUESTION_{i}: {question_text}")
        draw_number(int(eval(question_text)), canvas_x, canvas_y)
        pyautogui.press('enter')
        time.sleep(WAIT_DURATION)
    
    # pyautogui.moveTo(start_button_x, start_button_y, duration=MOVE_DURATION)
    # time.sleep(1)
    # pyautogui.click()

main()