from english_words import english_words_lower_alpha_set
import numpy as np
import cv2
from imutils import contours
import pyautogui
import time
import PIL.ImageGrab
import matplotlib.pyplot as plt

# w = wrong pos
# r = right pos

# For solving Wordle
def filter_by_guess(guess, words_array, list_valid_words):
    for i in range(0, 5):
        character = guess[i]
        if (character[0] == "n") and (character[2] not in list_valid_words):
            words_array = remove_with_character(character[2], words_array)
        elif character[0] == "w":
            words_array = must_have_character(character[2], i, words_array)
            list_valid_words.append(character[2])
        elif character[0] == "r":
            words_array = keep_character_with_position(character[2], i, words_array)
            list_valid_words.append(character[2])

    return words_array, list_valid_words


def remove_with_character(character, words_array):
    new_filter = []

    for item in words_array:
        if character in item:
            new_filter.append(False)
        else:
            new_filter.append(True)

    new_filter = np.array(new_filter)

    words_array = words_array[new_filter]

    return words_array


def must_have_character(character, pos, words_array):
    new_filter = []

    for item in words_array:
        if (character in item) and (item[pos] != character):
            new_filter.append(True)
        else:
            new_filter.append(False)

    new_filter = np.array(new_filter)
    words_array = words_array[new_filter]
    return words_array


def keep_character_with_position(character, pos, words_array):
    new_filter = []

    for item in words_array:
        if item[pos] == character:
            new_filter.append(True)
        else:
            new_filter.append(False)

    new_filter = np.array(new_filter)
    words_array = words_array[new_filter]
    return words_array

# Screen shots and proccessing

def get_keys_locations():
    print("Taking Screenshot Make Sure Wordle is full screen and minimize program")
    screen_resolution = [pyautogui.size()[0],  pyautogui.size()[1]]

    time.sleep(3)
    myScreenshot = pyautogui.screenshot()
    myScreenshot.save(r'screenshot.png')
    img = cv2.imread("screenshot.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Screenshot taken")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]

    # Find contours, sort from left-to-right, then crop
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Filter using contour area and extract ROI
    ROI_number = 0
    coordinates_bounding = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 10:
            x,y,w,h = cv2.boundingRect(c)
            ROI = img[y:y+h, x:x+w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
            ROI_number += 1
            coordinates_bounding.append([x+w/2,y+h/2])

    coordinates_bounding = np.array(coordinates_bounding)

    coordinates_bounding = coordinates_bounding[(coordinates_bounding[:, 1] > screen_resolution[1]*.74) &
                                                (coordinates_bounding[:, 1] < screen_resolution[1]*.95)]
    #plt.imshow(img)
    #plt.scatter(coordinates_bounding[:, 0], coordinates_bounding[:, 1])

    bottom_row = coordinates_bounding[coordinates_bounding[:, 1] < coordinates_bounding[:, 1].min() * 1.05]
    mid_val = (coordinates_bounding[:, 1].min() + coordinates_bounding[:, 1].max())/2
    middle_row = coordinates_bounding[(coordinates_bounding[:, 1] > mid_val * .95) & \
                                      (coordinates_bounding[:, 1] < mid_val * 1.05)]
    top_row = coordinates_bounding[coordinates_bounding[:, 1] > coordinates_bounding[:, 1].max() * .95]

    # plt.scatter(coordinates_bounding[:, 0], coordinates_bounding[:, 1])
    #
    # plt.scatter(top_row[:,0], top_row[:,1])

    bottom_row = bottom_row[bottom_row[:, 0].argsort()]
    middle_row = middle_row[middle_row[:, 0].argsort()]
    top_row = top_row[top_row[:, 0].argsort()]

    final_stack = np.vstack([bottom_row, middle_row, top_row])
    keys = ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p",
            "a", "s", "d", "f", "g", "h", "j", "k", "l",
            "*", "*", "*", "*", "*", "z", "x", "c", "v", "b", "n", "m", "del"]
    key_list = np.c_[keys]
    plt.scatter(final_stack[:, 0], final_stack[:, 1])

    keys_array_pos = np.hstack([final_stack, key_list])
    return keys_array_pos


def input_guess(guess, key_location):
    time.sleep(.5)
    for char in guess:
        pos_to_click = key_location[key_location[:, 2] == char]
        pyautogui.moveTo(float(pos_to_click[0][0]), float(pos_to_click[0][1]))
        time.sleep(.5)
        pyautogui.click()

    time.sleep(.5)
    pos_to_click = key_location[key_location[:, 2] == "*"][3]
    pyautogui.moveTo(float(pos_to_click[0]), float(pos_to_click[1]))
    time.sleep(.5)
    pyautogui.click()

def analyze_results(guess, key_location):
    output = []

    for i in range(0, 5):
        char = guess[i]
        pos_to_click = key_location[key_location[:, 2] == char]
        rgb = PIL.ImageGrab.grab().load()[float(pos_to_click[0][0]), float(pos_to_click[0][1])-25]

        if rgb == (106, 170, 100):
            output.append("r_" + char)

        if rgb == (120, 124, 126):
            output.append("n_" + char)

        if rgb == (201, 180, 88):
            output.append("w_" + char)

    return output

def check_if_done(results):
    num_correct = 0
    for item in results:
        if item[0] == "r":
            num_correct = num_correct + 1
    if num_correct == 5:
        return True

    return False

def main():
    words_array = np.array(list(english_words_lower_alpha_set))
    list_valid_words = []
    new_filter = []
    for item in words_array:
        if len(item) == 5 and item not in ["burtt", "armco", "perez", "berne", "percy", "jeres", "edith", "knott",
                                           "sudan", "shari", "deane", "beman", "byron", "carla", "paula", "siena", "boone", "mckee"]:
            new_filter.append(True)
        else:
            new_filter.append(False)

    new_filter = np.array(new_filter)
    words_array = words_array[new_filter]

    key_location = get_keys_locations()
    time.sleep(1)
    print("Starting Guessing")
    first_guess = "salet"

    input_guess(first_guess, key_location)
    time.sleep(3)
    results = analyze_results(first_guess, key_location)
    words_array, list_valid_words = filter_by_guess(results, words_array, list_valid_words)

    while True:
        # Bad test!!
        # if len(words_array) <= 1:
        #     break

        print(words_array)
        new_guess = words_array[0]
        time.sleep(.5)
        print(new_guess)
        input_guess(new_guess, key_location)
        time.sleep(3)
        results = analyze_results(new_guess, key_location)
        print(results)
        words_array, list_valid_words = filter_by_guess(results, words_array, list_valid_words)
        time.sleep(.5)


main()
