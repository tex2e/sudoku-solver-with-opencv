
import cv2
import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
import glob
import sys
import tensorflow as tf
from tensorflow.keras import layers, models
from sudokusolver import solve_sudoku

is_filter = True

def show(img, title="", destroy=True):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()


imgs = list(glob.glob('data/img/*.JPG'))
imgs.sort()
img_index = 0 if len(sys.argv) < 2 else int(sys.argv[1])
file_path = imgs[img_index]

WIDTH, HEIGHT = 576, 576  # 64 * 9 = 576
SHAPE = (WIDTH, HEIGHT)

orig_img = cv2.imread(file_path)
img = orig_img.copy()
img = cv2.resize(img, SHAPE)
img1 = img.copy()
img2 = img.copy()
img3 = img.copy()
print(img.shape)
show(img)

# https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python#answers

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# アンシャープマスク
kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]], np.float32)
gray = cv2.filter2D(gray, -1, kernel)
# エッジ検出
edges = cv2.Canny(gray, 100, 200, apertureSize=3)
show(edges)
# 膨張
kernel = np.ones((5,5), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
show(edges)
# 浸食
kernel = np.ones((7,7), np.uint8)
edges = cv2.erode(edges, kernel, iterations=1)
show(edges)
# cv2.imwrite('img1.jpg', edges)

lines = cv2.HoughLines(edges, 0.5, np.pi/180, 150)
# print(lines)

if not lines.any():
    print('No lines were found')
    exit()

if is_filter:
    rho_threshold = 40
    theta_threshold = 0.2

    # how many lines are similar to a given one
    similar_lines = {i : [] for i in range(len(lines))}
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue

            rho_i,theta_i = lines[i][0]
            rho_j,theta_j = lines[j][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                similar_lines[i].append(j)

    # ordering the INDECES of the lines by how many are similar to them
    indices = [i for i in range(len(lines))]
    indices.sort(key=lambda x: len(similar_lines[x]))

    # line flags is the base for the filtering
    line_flags = [True] * len(lines)
    for i in range(len(lines) - 1):
        # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
        if not line_flags[indices[i]]:
            continue

        # we are only considering those elements that had less similar line
        for j in range(i + 1, len(lines)):
            # and only if we have not disregarded them already
            if not line_flags[indices[j]]:
                continue

            rho_i,theta_i = lines[indices[i]][0]
            rho_j,theta_j = lines[indices[j]][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                # if it is similar and have not been disregarded yet then drop it now
                line_flags[indices[j]] = False

print('number of Hough lines:', len(lines))

filtered_lines = []

if is_filter:
    for i in range(len(lines)): # filtering
        if line_flags[i]:
            filtered_lines.append(lines[i])

    print('Number of filtered lines:', len(filtered_lines))
else:
    filtered_lines = lines


# 直線を検出する
lines_xy = []

for line in filtered_lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    lines_xy.append((x2, y2, x1, y1))

    cv2.line(img1, (x1,y1), (x2,y2), (0,0,255), 2)

show(img1)

# cv2.imwrite('res/f%02d-lines.jpg' % (img_index), img1)


# 直線の交点を求める

def line_intersect(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):
    """ returns a (x, y) tuple or None if there is no intersection """
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return None, None
    if not(0 <= uA <= 1 and 0 <= uB <= 1):
        return None, None
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)
    return int(x), int(y)

intersections = []
for line1, line2 in itertools.combinations(lines_xy, 2):
    Ax1, Ay1, Ax2, Ay2 = line1
    Bx1, By1, Bx2, By2 = line2

    x, y = line_intersect(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2)
    if x is None:
        continue
    if 0 < x < WIDTH and 0 < y < HEIGHT:
        intersections.append((x, y))


# 左上、左下、右上、右下の交点を求める

def distance(A, B):
    Ax, Ay = A
    Bx, By = B
    return math.sqrt((Ax - Bx)**2 + (Ay - By)**2)

ABOVE_LEFT = (0, 0)
most_above_left = (distance(intersections[0], ABOVE_LEFT), intersections[0])
BELOW_LEFT = (0, HEIGHT)
most_below_left = (distance(intersections[0], BELOW_LEFT), intersections[0])
ABOVE_RIGHT = (WIDTH, 0)
most_above_right = (distance(intersections[0], ABOVE_RIGHT), intersections[0])
BELOW_RIGHT = (WIDTH, HEIGHT)
most_below_right = (distance(intersections[0], BELOW_RIGHT), intersections[0])

for point in intersections[1:]:
    dist = distance(point, ABOVE_LEFT)
    if dist < most_above_left[0]:
        most_above_left = (dist, point)
    dist = distance(point, BELOW_LEFT)
    if dist < most_below_left[0]:
        most_below_left = (dist, point)
    dist = distance(point, ABOVE_RIGHT)
    if dist < most_above_right[0]:
        most_above_right = (dist, point)
    dist = distance(point, BELOW_RIGHT)
    if dist < most_below_right[0]:
        most_below_right = (dist, point)

img2 = cv2.circle(img2, most_above_left[1],  10, (0, 0, 255), -1) # red circle
img2 = cv2.circle(img2, most_below_left[1],  10, (0, 0, 255), -1) # red circle
img2 = cv2.circle(img2, most_above_right[1], 10, (0, 0, 255), -1) # red circle
img2 = cv2.circle(img2, most_below_right[1], 10, (0, 0, 255), -1) # red circle
show(img2)


# 射影変換

pts1 = np.float32([most_above_left[1],  most_below_left[1],
                   most_above_right[1], most_below_right[1]])
pts2 = np.float32([ABOVE_LEFT, BELOW_LEFT, ABOVE_RIGHT, BELOW_RIGHT])

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(img3, M, SHAPE)
show(dst)

# cv2.imwrite('res/f%02d-trans.jpg' % (img_index), dst)


# 数字部分の切り取り (64x64)

size = WIDTH // 9

matrix = []
for y in range(9):
    row = []
    for x in range(9):
        row.append(dst[y*size:(y+1)*size, x*size:(x+1)*size])
    matrix.append(row)

# print(matrix[0][0].shape)


# 数字の認識

model = models.load_model('my_digit_model.h5')

digit_results = []

plt.figure(figsize=(10, 10))
plt.subplots_adjust(hspace=0.5)
for y in range(9):
    for x in range(9):
        img = matrix[y][x]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

        # # 数字の画像を保存する (機械学習の訓練用)
        # cv2.imwrite('data/digits/f%02d-n%02d-d_.jpg' % (img_index, y*9 + x + 1), img)

        img = cv2.bitwise_not(img)
        target = np.array([img])
        target = target.reshape(target.shape + (1,))
        result = model.predict(target)
        predicted_class = np.argmax(result[0], axis=-1)

        digit_results.append(predicted_class)

        plt.subplot(9, 9, y*9 + x + 1)
        plt.axis('off')
        plt.title(predicted_class)
        plt.imshow(img, 'gray')

plt.show()


# Z3で数独問題を解く

problem = np.array(digit_results, np.uint8).reshape(9, 9).tolist()

solve_sudoku(problem)
