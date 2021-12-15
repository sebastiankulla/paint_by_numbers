import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
from typing import Optional


def flatten_image(image_array: np.ndarray) -> np.ndarray:
    """
    transform a color image with shape (height, width, rgb) to (height*width, rgb)
    :param image_array: numpy array of rgb image
    :return: numpy array of flattened image
    """
    height = image_array.shape[0]
    width = image_array.shape[1]
    return np.reshape(image_array, (height * width, 3))


def load_image(path_pic: str) -> np.ndarray:
    """
    loads rgb image of jpg format to numpy array
    :param path_pic: path to image you want to load
    :return: image data in a numpy array
    """
    return cv2.cvtColor(cv2.imread(path_pic), cv2.COLOR_BGR2RGB)


def save_image(filename: str, image_array: np.ndarray) -> None:
    """
    Saves image array to file
    :param filename: path and name of the image you want to save
    :param image_array: image data in a numpy array
    """
    cv2.imwrite(filename, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))


def reshape_image(flattened_image_array: np.ndarray, height: int, width: int, shrink: bool = False) -> np.ndarray:
    """

    :param flattened_image_array:
    :param height:
    :param width:
    :param shrink:
    :return:
    """
    temp_array = np.reshape(flattened_image_array, (height, width, -1))
    if shrink:
        return temp_array[:, :, 0]
    else:
        return temp_array


def show_image(image_array: np.ndarray) -> None:
    """
    Shows the image data with matplotlib
    :param image_array: numpy array of the image data you want to show
    """
    fig, ax1 = plt.subplots()
    ax1.imshow(image_array)


def clean_image(image_array: np.ndarray, filter_strength: int = 14, template_window_size: int = 7,
                search_window_size: int = 21) -> np.ndarray:
    """
    Denoises the image with fastNLMeansDenoising and the morphological transformations opening and closing
    :param image_array: numpy array of the image
    :param filter_strength: the strenght of the means denoising
    :param template_window_size:
    :param search_window_size:
    :return: returns denoised image in a numpy array
    """
    """Reduction of noize, Morphomat operations, opening then closing """
    clean_pic = cv2.fastNlMeansDenoisingColored(image_array, None, filter_strength, filter_strength,
                                                template_window_size, search_window_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    clean_pic = cv2.morphologyEx(clean_pic, cv2.MORPH_OPEN, kernel, cv2.BORDER_REPLICATE)
    clean_pic = cv2.morphologyEx(clean_pic, cv2.MORPH_CLOSE, kernel, cv2.BORDER_REPLICATE)
    return clean_pic


def replace_colors(labeled_image_array: np.ndarray, color_array: np.ndarray) -> np.ndarray:
    """
    wird benutzt um die labels, die von kmean.labels ausgeben werden mit den Farbwerten von kmean.cluster_centers_ zu ersetzen
    :param labeled_image_array: numpy array of the image with every pixel clustered into a label (shape=(height, width, 1))
    :param color_array:
    :return:
    """
    new_array = np.zeros_like(labeled_image_array)
    for i in range(np.max(labeled_image_array) + 1):
        new_array = np.where(labeled_image_array == i, color_array[i].astype(int), new_array)
    return new_array


def euclidean_distance(from_color: np.ndarray, to_color: np.ndarray) -> np.ndarray:
    """
    the function calculates the distance between the pixels of an image and a predefined color
    :param from_color: numpy array of an image
    :param to_color: array of RGB values to which the distance is to be calculated
    :return: numpy array of distances to predefined color
    """
    return np.sum(np.power(from_color - to_color, 2), axis=2)


def calc_distance_to_colors(image_array: np.ndarray, color_list: np.ndarray) -> np.ndarray:
    """
    Calculates the distance from the color value of the image to each color value in the color_list and
    returns which layer has the smallest distance to the color value.
    :param image_array: numpy array of an image
    :param color_list: Array of multiple RGB values
    :return: Returns an image sized array with the index of the color that has the shortest distance
    """
    count_values = len(color_list)
    new_array = np.zeros((image_array.shape[0], image_array.shape[1], count_values))
    for idx, value in enumerate(color_list):
        new_array[:, :, idx] = euclidean_distance(image_array, value)
    return np.argmin(new_array, axis=2)[:, :, np.newaxis]


def draw_contours(image_array: np.ndarray, color_list: np.ndarray, min_sized_contour: int = 20,
                  colored_lines: bool = False, add_label: bool = True, show_colored: bool = False) -> np.ndarray:
    """
    Draws a new image with contours and optional contour color label
    :param image_array: numpy array of an image
    :param color_list: array of colors used in the image
    :param min_sized_contour: The minimal quadratic pixel size of a contour shown
    :param colored_lines: defines if the contours should have a colored border
    :param add_label: defines if there should by labels with the index of the used color in each contour
    :param show_colored: defines if the image should be returned completely colord, like it's already painted
    :return: returns image array of the paint by number image
    """
    canvas = np.ones_like(image_array, dtype="uint8") * 255
    for ind, color in enumerate(color_list):
        mask = cv2.inRange(image_array, color, color)
        cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for contour in cnts:
            _, _, width_ctr, height_ctr = cv2.boundingRect(contour)
            if width_ctr > min_sized_contour and height_ctr > min_sized_contour and cv2.contourArea(contour,
                                                                                                    True) < - 100:
                if colored_lines:
                    line_color = color.astype(int).tolist()
                else:
                    line_color = [0, 0, 0]

                cv2.drawContours(canvas, [contour], -1, line_color, 1)

                if show_colored:
                    cv2.fillPoly(canvas, pts=[contour], color=color.astype(int).tolist())
                if add_label:
                    # txt_x, txt_y = ((np.max(contour, axis=(0,1)) + np.min(contour, axis=(0,1)))/2).astype(int)
                    M = cv2.moments(contour)
                    txt_x = int(M["m10"] / M["m00"])
                    txt_y = int(M["m01"] / M["m00"])
                    cv2.putText(canvas, '{:d}'.format(ind + 1),
                                (txt_x - 5, txt_y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
    return canvas


def relabel_image(labeled_image_array: np.ndarray, color_list: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Based on a labeled image, the color palette used is reduced.
    The labels are adjusted so that they start from 0.
    :param labeled_image_array: numpy array of a labeled image
    :param color_list: array of colors used in the image
    :return: relabeled image array and an adjusted color_list
    """
    new_array = np.ones_like(labeled_image_array)
    new_color_list = np.ones((len(np.unique(labeled_image_array)), 3))
    for idx, value in enumerate(np.unique(labeled_image_array)):
        new_array = np.where(labeled_image_array == value, idx, new_array)
        new_color_list[idx, :] = color_list[value]

    return new_array, new_color_list


def create_image_by_kmeans(image_file: str, number_of_colors: int, min_sized_contour: int = 20,
                           filter_strength: int = 14, colored_lines=False, add_label=True, show_colored=False,
                           output_file: Optional[str] = None):
    """

    :param image_file:
    :param number_of_colors:
    :param min_sized_contour:
    :param filter_strength:
    :param colored_lines:
    :param add_label:
    :param show_colored:
    :param output_file:
    :return:
    """
    if output_file is None:
        output_file = image_file.lower().split('.jpg')[0] + '_paintbynumbers.jpg'

    image_array = load_image(image_file)
    cleaned_image_array = clean_image(image_array, filter_strength=filter_strength)
    height, width, _ = cleaned_image_array.shape
    flattened_image_array = flatten_image(cleaned_image_array)
    kmeans = KMeans(n_clusters=number_of_colors, random_state=0).fit(flattened_image_array)
    reshaped_image = reshape_image(kmeans.labels_, height, width)
    new_color_image = replace_colors(reshaped_image, kmeans.cluster_centers_)
    contoured_image = draw_contours(new_color_image, kmeans.cluster_centers_, min_sized_contour, colored_lines,
                                    add_label, show_colored)
    save_image(output_file, contoured_image)
    return contoured_image


def create_image_by_predefined_colors(image_file: str, color_list: np.ndarray, min_sized_contour: int = 20,
                                      filter_strength: int = 14, colored_lines=False, add_label=True,
                                      show_colored=False, output_file: Optional[str] = None):
    """

    :param image_file:
    :param color_list:
    :param min_sized_contour:
    :param filter_strength:
    :param colored_lines:
    :param add_label:
    :param show_colored:
    :param output_file:
    :return:
    """
    if output_file is None:
        output_file = image_file.lower().split('.jpg')[0] + '_paintbynumbers.jpg'

    image_array = load_image(image_file)
    cleaned_image_array = clean_image(image_array, filter_strength=filter_strength)
    labeled_image_array = calc_distance_to_colors(cleaned_image_array, color_list)
    relabeled_image_array, new_color_list = relabel_image(labeled_image_array, color_list)
    new_color_image = replace_colors(relabeled_image_array, new_color_list)
    contoured_image = draw_contours(new_color_image, new_color_list, min_sized_contour, colored_lines, add_label,
                                    show_colored)

    save_image(output_file, contoured_image)
    return contoured_image


class ImageConverter:
    def __init__(self, image_file: str, min_sized_contour: int = 20, filter_strength: int = 14,
                 colored_lines=False, add_label=True, show_colored=False, output_file: Optional[str] = None):
        self.image_file = image_file
        self.min_sized_contour = min_sized_contour
        self.filter_strength = filter_strength
        self.colored_lines = colored_lines
        self.add_label = add_label
        self.show_colored = show_colored
        self.output_file = output_file if output_file is not None else image_file.lower().split('.jpg')[
                                                                           0] + '_paintbynumbers.jpg'
        self.template_window_size = 7
        self.search_window_size = 21

    def create_image_by_predefined_colors(self, color_list):
        self.real_image_array = load_image(self.image_file)
        image_array = self.clean_image(self.real_image_array)
        image_array = self.replace_color_with_color_list(image_array, color_list)
        image_array, color_list = self.relabel_image(image_array, color_list)
        image_array = replace_colors(image_array, color_list)

        contours = self.find_contours(image_array, color_list)
        self.contours = self.filter_contours(contours)
        self.new_image = self.draw_contours(image_array)

    def clean_image(self, image_array):
        clean_pic = cv2.fastNlMeansDenoisingColored(image_array, None, self.filter_strength, self.filter_strength,
                                                    self.template_window_size, self.search_window_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        clean_pic = cv2.morphologyEx(clean_pic, cv2.MORPH_OPEN, kernel, cv2.BORDER_REPLICATE)
        clean_pic = cv2.morphologyEx(clean_pic, cv2.MORPH_CLOSE, kernel, cv2.BORDER_REPLICATE)
        return clean_pic

    def replace_color_with_color_list(self, image_array, color_list):
        count_values = len(color_list)
        new_array = np.zeros((image_array.shape[0], image_array.shape[1], count_values))
        for idx, value in enumerate(color_list):
            new_array[:, :, idx] = euclidean_distance(image_array, value)

        return np.argmin(new_array, axis=2)[:, :, np.newaxis]

    def relabel_image(self, image_array, color_list):
        new_array = np.ones_like(image_array)
        new_color_list = np.ones((len(np.unique(image_array)), 3))
        for idx, value in enumerate(np.unique(image_array)):
            new_array = np.where(image_array == value, idx, new_array)
            new_color_list[idx, :] = color_list[value]
        return new_array, new_color_list

    def find_contours(self, image_array, color_list):
        all_cnts = []
        for ind, color in enumerate(color_list):
            mask = cv2.inRange(image_array, color, color)
            temp_cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            temp_cnts = temp_cnts[0] if len(temp_cnts) == 2 else temp_cnts[1]
            temp_cnts = [Contour(cnt, color, ind + 1) for cnt in temp_cnts]
            all_cnts = all_cnts + temp_cnts
        return all_cnts

    def filter_contours(self, contours):
        filter_func = lambda x: x.area > 1500 and x.width > self.min_sized_contour and x.height > self.min_sized_contour
        filtered_contours = filter(filter_func, contours)
        filtered_contours = sorted(filtered_contours, key=lambda x: x.area, reverse=True)
        return filtered_contours

    def draw_contours(self, image_array):
        canvas = np.ones_like(image_array, dtype="uint8") * 255
        for contour in self.contours:
            contour.draw_to_image_array(canvas, self.colored_lines)
            if self.show_colored:
                contour.fill_contour_image_array(canvas)
            if self.add_label:
                contour.put_label_image_array(canvas)

        return canvas

    def save_image(self, output_file=None):
        if output_file is None:
            output_file = self.image_file.lower().split('.jpg')[0] + '_paintbynumbers.jpg'
        if not output_file.endswith('.jpg'):
            output_file += '.jpg'
        save_image(output_file, self.new_image)

    def save_svg(self, output_file=None):
        if output_file is None:
            output_file = self.image_file.lower().split('.jpg')[0] + '_paintbynumbers.svg'
        if not output_file.endswith('.svg'):
            output_file += '.svg'
        SvgFile(self.contours).save_file(output_file)



class Contour:
    def __init__(self, contour, color=None, color_id=None):
        self.contour = contour
        self.width, self.height, self.txt_x, self.txt_y = 0, 0, 0, 0
        self.area = abs(cv2.contourArea(contour, True))
        self.color = color
        self.color_id = color_id
        if self.area > 0:
            _, _, self.width, self.height = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            self.txt_x = int(M["m10"] / M["m00"])
            self.txt_y = int(M["m01"] / M["m00"])
            # txt_x, txt_y = ((np.max(contour, axis=(0,1)) + np.min(contour, axis=(0,1)))/2).astype(int)

    def draw_to_image_array(self, image_array, colored_line=False):
        line_color = self.color.astype(int).tolist() if colored_line else [0, 0, 0]
        cv2.drawContours(image_array, [self.contour], -1, line_color, 1)

    def put_label_image_array(self, image_array):
        cv2.putText(image_array, '{:d}'.format(self.color_id),
                    (self.txt_x - 5, self.txt_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)

    def fill_contour_image_array(self, image_array):
        cv2.fillPoly(image_array, pts=[self.contour], color=self.color.astype(int).tolist())

    # def draw_to_svg(self, svg_file_stream)

    def __str__(self):
        return "<Contour Width: {}, Height: {}, Area: {}, Color: {}, ColorID: {}>".format(self.width, self.height,
                                                                                          self.area, self.color,
                                                                                          self.color_id)

    def __repr__(self):
        return "<Contour Width: {}, Height: {}, Area: {}, Color: {}, ColorID: {}>".format(self.width, self.height,
                                                                                          self.area, self.color,
                                                                                          self.color_id)


class SvgFile:
    def __init__(self, contours):
        self.contours = contours
        self.header = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                       '<svg xmlns="http://www.w3.org/2000/svg"\n'
                       'xmlns:xlink="http://www.w3.org/1999/xlink"\n'
                       'version="1.1" baseProfile="full">\n\n')
        self.footer = '</svg>'
        self.styles = ('<style>\n'
                       '.small { font: italic 13px sans-serif; }\n'
                       '.heavy { font: bold 30px sans-serif; }\n'
                       '.Rrrrr { font: italic 40px serif; fill: red; }\n'
                       '</style>\n\n')
        self.paths = []
        self.texts = []
        self.content = self.create_content()

    @staticmethod
    def _contour_to_path(contour):
        squeezed_array = np.squeeze(contour.contour, axis=1)

        temp_path = ''
        temp_path += '<path fill="none" stroke="black" d="'
        temp_path += 'M {},{} '.format(squeezed_array[0][0], squeezed_array[0][1])

        for row in squeezed_array[1:]:
            temp_path += 'L {},{} '.format(row[0], row[1])

        temp_path += 'Z" />\n'
        return temp_path

    @staticmethod
    def _contour_to_text(contour):
        temp_text = '<text x="{}" y="{}" class="small">{}</text>\n'.format(contour.txt_x, contour.txt_y,
                                                                           contour.color_id)
        return temp_text

    def add_paths(self):
        paths = []
        for contour in self.contours:
            paths.append(self._contour_to_path(contour))
        return paths

    def add_texts(self):
        texts = []
        for contour in self.contours:
            texts.append(self._contour_to_text(contour))
        return texts

    def create_content(self):
        paths = self.add_paths()
        texts = self.add_texts()

        temp_content = ''
        temp_content += self.header
        temp_content += self.styles

        for path in paths:
            temp_content += path

        for text in texts:
            temp_content += text

        temp_content += self.footer

        return temp_content

    def save_file(self, filename=None):
        if filename is None:
            filename = 'export.svg'

        if not filename.endswith('.svg'):
            filename += '.svg'

        with open(filename, 'w') as file:
            file.write(self.content)
