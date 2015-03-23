# coding=utf-8
# coding=utf-8
# coding=utf-8
# coding=utf-8
# coding=utf-8
__author__ = 'myachikov'

import cv2
import numpy as np
import cmath
from copy import copy
from itertools import combinations

def all_distinct(sets):
    for i, j in combinations(sets, r=2):
        if i.intersection(j):
            return False
    return True

def merge(sets):
    while not all_distinct(sets):
        for a, b in combinations(sets, r=2):
            if a.intersection(b):
                c = a.union(b)
                if a in sets:
                    sets.remove(a)
                if b in sets:
                    sets.remove(b)
                sets.append(c)
    return sets


def areas_depth_first(image, min_pixels=10):
    height, width = image.shape

    tmp_image = copy(image)
    connected = np.zeros_like(image)

    objects = 0
    regions = {}

    def neighbours(coors):
        i, j = coors
        ns = [(i, j-1), (i, j+1), (i-1, j), (i+1, j)]
        return [(k,l) for k,l in ns if 0 <= k < height and 0 <= l < width and tmp_image[k][l] > 0 and connected[k][l] == 0]

    def fill_neighbours(coors, label):
        pixels = [coors]
        pixels_count = 0
        regions[label] = []
        while pixels != []:
            current_pixel = pixels.pop()
            regions[label].append(current_pixel)
            pixels_count += 1
            pixels += neighbours(current_pixel)
            k, l = current_pixel
            connected[k][l] = label
            tmp_image[k][l] = 0

        if pixels_count < min_pixels:
            for i,j in regions[label]:
                connected[i][j] = 0
            regions.pop(label)

        return pixels_count

    for i in xrange(height):
        for j in xrange(width):
            if tmp_image[i][j] > 0:
                objects += 1
                pixels_count = fill_neighbours((i,j), objects)

    return connected

def neighbours(coors):
        i, j = coors
        return [(i, j-1), (i-1, j-1), (i-1, j), (i-1, j+1), (i, j+1), (i+1, j+1), (i+1, j), (i+1, j-1)]


def areas_two_pass(image, minpixels=10):
    height, width = image.shape

    connected = np.zeros_like(image)

    objects = 0
    regions = {}
    equivalences = []
    occur = []

    for i in xrange(height):
        for j in xrange(width):

            current_value = image[i][j]
            if current_value:
                west_value = image[i][j-1] if j >= 1 else 0
                west_label = connected[i][j-1] if j >= 1 else 0
                north_value = image[i-1][j] if i >= 1 else 0
                north_label = connected[i-1][j] if i >= 1 else 0
                if west_value != current_value and north_value != current_value:
                    objects += 1
                    connected[i][j] = objects

                elif west_value != current_value and north_value == current_value:
                    connected[i][j] = north_label

                elif west_value == current_value and north_value != current_value:
                    connected[i][j] = west_label

                elif west_value == current_value and north_value == current_value:
                    connected[i][j] = north_label
                    if west_label != north_label and not {west_label, north_label} in equivalences:
                        equivalences.append({west_label, north_label})
                        merge(equivalences)

    merge(equivalences)

    label_map = range(objects + 1)

    for eq in equivalences:
        min_element = min(eq)
        for el in eq:
            label_map[el] = min_element
    for i in xrange(height):
        for j in xrange(width):
            connected[i][j] = label_map[connected[i][j]]

    regions = {}
    for i in xrange(height):
        for j in xrange(width):
            current_label = connected[i][j]
            if current_label in regions:
                regions[current_label].append((i, j))
            else:
                regions[current_label] = [(i, j)]
    for label, region in regions.items():
        if len(region) < minpixels:
            for i, j in region:
                connected[i][j] = 0

    return connected

def contours_moore(image, minlength=None):

    def label(coors):
        i, j = coors
        if i < 0 or j < 0 or i >= height or j >= width:
            return None
        else:
            return image[i][j]

    def contour(start, current_label):
        pixels = [start]
        i, j = start
        enter_from = (i-1, j)
        next_pixel = (-1,-1)
        while next_pixel != start:
            ns = neighbours(pixels[-1])
            d = ns.index(enter_from)
            ns = ns[d:] + ns[:d]
            ns_labels = map(label, ns)
            if current_label in ns_labels:
                pixel_with_label_index = ns_labels.index(current_label)
            else:
                break
            next_pixel = ns[pixel_with_label_index]
            enter_from = ns[pixel_with_label_index-1]
            pixels.append(next_pixel)

        pixels = [(j, i) for i, j in pixels]
        pixels = map(np.array, pixels)

        return np.array(pixels)

    height, width = image.shape[:2]

    labels = []
    contours = []

    for i in xrange(height):
        for j in xrange(width):
            current_label = image[i][j]
            if current_label != 0:
                if not (current_label in labels):
                    contours.append(contour((i, j), current_label))
                    labels.append(current_label)
    if not minlength is None:
        contours = [contour for contour in contours if len(contours) >= minlength]
    return tuple(contours)

def contour_coors_to_complex(contour):
    result = [complex(0, 0) for _ in xrange(len(contour))]
    for i in xrange(len(contour)):
        j = (i + 1) % len(contour)
        x, y = contour[j] - contour[i]
        result[i] = complex(x, y)

    return result

if __name__ == '__main__':
    test = cv2.imread('test.tiff')[:, :, 1]
    #test = np.array([[0,1,1,0,0,1], [1,1,1,1,0,1], [0,1,0,0,0,1], [0,0,0,1,1,1], [0,0,0,1,1,1]])
    test_out = areas_two_pass(test, 10)
    contours = contours_moore(test_out)
    contours = [contour_coors_to_complex(contour) for contour in contours]
    cv2.imwrite('test_out.tiff', test_out * 10)