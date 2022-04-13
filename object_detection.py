
import pandas as pd
from tqdm import tqdm
import os
from google.cloud import vision
from google.cloud.vision_v1 import types
from google.oauth2 import service_account
import requests
import logging
from io import BytesIO
from collections import Counter


class Logger:
    """
    I copied this logger from some other script I once found, I like to include
    it. It's more common to include this logger in its own module in your
    package but for now that is unnecessary.
    I make an instance of the logger inside the other classes to log what is is
    doing.
    """

    def __init__(self, level=logging.INFO):
        logfmt = "[%(asctime)s - %(levelname)s] - %(message)s"
        dtfmt = "%Y-%m-%d %I:%M:%S"
        logging.basicConfig(level=level, format=logfmt, datefmt=dtfmt)

    def get(self):
        return logging.getLogger()


class ImageLabeler:
    """
    Tool to get 10 labels per photo using Google Cloud Vision
    """

    def __init__(self, auth_file, verbose=True):
        """

        :param auth_file: A JSON file with credentials associated with your
        Google account. You can create a file like this on console.cloud.google.com.
        Make sure you enable Google Cloud Vision API and create a new service account
        with credentials to create this JSON file. For help see
        https://cloud.google.com/vision/docs/setup

        :param verbose: If you set this to False the logger will only show error
        messages and skip the info messages
        """

        if not verbose:
            lg = Logger(level=logging.ERROR)
        else:
            lg = Logger()
        self.__logger = lg.get()

        credentials = service_account.Credentials.from_service_account_file(auth_file)
        self.client = vision.ImageAnnotatorClient(credentials=credentials)

    def _load_image(self, photo_url):
        try:
            # Load the image into a Google format
            response = requests.get(photo_url)
            with BytesIO(response.content) as image_file:
                content = image_file.read()
            image = types.Image(content=content)

        except Exception as err:
            self.__logger.error(f"Error retrieving image from {photo_url}: {err}")
            image = None

        return image

    def get_label_from_image(self, photo_url):
        image = self._load_image(photo_url)

        try:
            # label the image
            response = self.client.label_detection(image=image)
            labels = response.label_annotations
        except Exception as err:
            self.__logger.error(f"Error extracting labels from Google Vision: {err}")
            labels = []  # return empty label list when error occurs
                         # Now your process won't crash if some unexpected error
                         # occurs

        return labels


def get_url(url_sq):
    """
    For some reason I had problems getting the right link when downloading
    form the Flickr API. This function translates the thumbnail URL that
    I use from Flickr into the URl of the actual size photo.
    It removes the "_s" from the URL.
    :param url_sq: Thumbnail URL (ending with the "_s.jpg"
    :return:
    """
    try:
        url = url_sq[:len(url_sq) - 6] + '.jpg'
    except Exception as err:
        url = 'unknown'
    return url


class DfLabeler:
    """
    This tool applies the ImageLabeler tool to a csv dataset with a column
    of urls.
    """

    def __init__(self, data_csv, auth_file, url_col, url_sq=False):
        """

        :param data_csv: path to csv with photos containing a column with urls
        :param auth_file: json with Google Vision credentials, see
        https://cloud.google.com/vision/docs/setup
        :param url_col: exact name of column in csv with urls
        :param url_sq: if the url column is a Flickr url with "_s" keep True
        so it will edit the url to get the right url. Otherwise use False
        """
        self.data_df = pd.read_csv(data_csv)
        self.labeler = ImageLabeler(auth_file)
        lg = Logger()
        self.__logger = lg.get()
        if url_sq:
            self.urls = [get_url(url_sq) for url_sq in self.data_df[url_col]]  # change url
        else:
            self.urls = list(self.data_df[url_col])  # use the actual url in csv
        self.df = []
        self.labels = []

    def get_labels_from_df(self):
        """
        get the labels for all urls in csv. This can take a while
        :return: List of dictionaries with labels and confidence per photo
        """
        cost = len(self.urls) // 1000 * 1.5  # estimate the costs charged by Google
        self.__logger.info(f"Labeling {len(self.urls)} photos with Google Cloud Vision API \n"
                           f"Total costs: ${cost}")  # please check Google Cloud console for actual costs charged
        self.labels = [self.labeler.get_label_from_image(url) for url in tqdm(self.urls)]
        return self.labels

    def store_labels(self, labels):
        """
        Change the output from  get_labels_from_df to a list of dictionairies
        that we can add to the original csv
        :param labels: The list of labels created in get_labels_from_df
        :return: A prettier list of dictionairies with labels and confidence
        """
        for photos in labels:
            label_list = [photo.description for photo in photos]
            scores = [photo.score for photo in photos]
            label_cols = {f"label_{i + 1}": label_list[i] for i in range(len(label_list))}
            score_cols = {f"score_{i + 1}": scores[i] for i in range(len(scores))}
            label_cols.update(score_cols)
            self.df.append(label_cols)
        return self.df

    def update_data_df(self, df, output_path):
        """
        Update the original csv with the labels and confidence per photo
        :param df: list of dictionaires created in store_labels
        :param output_path: path to new csv
        :return: -
        """
        new_df = pd.DataFrame(df)
        a = self.data_df.join(new_df)
        a.to_csv(output_path)

    def get_label_counts(self, output_path):
        """
        Get all unique words in database with the total counts per word
        :param output_path: path to store the csv with words and counts
        :return:
        """
        # get list of all words
        all_words = []
        for i in tqdm(self.df):
            vals = list(i.values())
            words = [elm for elm in vals if isinstance(elm, str)]
            all_words.extend(words)

        # get count of every unique word in list
        new_vals = Counter(all_words).most_common()

        # Export
        word_list = [{'word': x, 'count': y} for x, y in new_vals]
        word_df = pd.DataFrame(word_list)
        word_df.to_csv(output_path)
        return word_df

if __name__ == '__main__':
    """
    Run the script with your own parameters. If you import the code as a module
    these would be the commands you have to run
    """

    # Define working folder
    os.chdir('C:/users/tieskens/dropbox/wolfs/haaien')

    # path to Google credentials JSON
    auth_file = 'google_key/zandmotor-288919-b347f422030c.json'

    # path to csv with data with photos contianing url column
    data_csv = 'output/maarten_flickr_simple.csv'

    # path to csv that will be created with labels added
    output_path = 'data/maarten_labeled.csv'
    output_path_words = 'data/maarten_words.csv'

    # run the labeler

    lab = DfLabeler(data_csv, auth_file, url_col='url_sq', url_sq=True)
    labels = lab.get_labels_from_df()
    df = lab.store_labels(labels)
    lab.update_data_df(df, output_path)
    lab.get_label_counts(output_path_words)









