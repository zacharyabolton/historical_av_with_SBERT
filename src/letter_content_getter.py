import sys
import os
from bs4 import BeautifulSoup


def get_text_content(content_dir):
    """
    Gets the text content from all of The Valley of the Shadow Augusta
    Virginia letters (link below) dataset and stores them as the value in a
    key value pair, where the key is a ref back to the original raw data.
    For more on The Valley of the Shadow dataset see Dataset_Card.md
    Returns a dataframe where rows represent letters, and columns are
    metadata fields, with the new `author` field appended to the end.

    https://valley.newamericanhistory.org/search/letters/results?county=augusta

    :param content_dir: full path to the directory containing the html
    content files.
    :type content_dir: string
    :rtype dict:
    """
    contents = dict()
    for file in os.listdir(content_dir):
        if file.endswith('.html'):
            with open(os.path.join(content_dir, file)) as hc:
                soup = BeautifulSoup(hc, "html.parser")
                # Adapted from code by Theo Vasilis
                # found at https://blog.apify.com/beautifulsoup-find-by-class/
                body = soup.find_all(class_='source__body')
                content = ''.join(
                    body[i].get_text()
                    for i
                    in range(len(body)))
                contents[file[:-5]] = content
    return contents