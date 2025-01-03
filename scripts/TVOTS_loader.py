"""
Processes files obtained by `the_valley_of_the_shadow_downloader.py` script.
Takes a path to the directory output by
`the_valley_of_the_shadow_downloader.py` and produces a new directory in
'data/cleaning_in_process/imposters/tvots' with a lowercased, snake_cased,
middle name removed, subdirectory per unique author found.

E.g. If the source directory contains letters from John B Good, and Jane B
Good, this script produces directories for their respective letters named
'john_good' and 'jane_good'.
This method risks combining works from different authors, but has been deamed
acceptible.
"""
import json
import sys
import os
import pandas as pd
from bs4 import BeautifulSoup
from definitions import ROOT_DIR

# Add src directory to sys.path
# Adapted from Taras Alenin's answer on StackOverflow at:
# https://stackoverflow.com/a/55623567
src_path = os.path.join(ROOT_DIR, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)


#############################################################################
# Helper functions
#############################################################################

# Adapted from https://www.w3schools.com/python/pandas/ref_df_apply.asp
def remove_middle_name(s):
    """
    Takes in a name of unknown length and outputs a snake_case version of the
    name.
    If the name is three in length (John B Good), it removes the middle name,
    otherwise it does nothing additionally.

    :param s: string representing a name
    :type s: string
    :rtype string:
    """
    names = s.split()
    if len(names) == 3:
        return str(names[0]) + '_' + str(names[2])
    else:
        return '_'.join(names)


def get_unique_author_strings_and_text_content(data_dir):
    """
    - Augments The Valley of the Shadow Augusta Virginia letters (link below)
      dataset with a field for the author as represented by a transformed
      snake_case string.
    - Culls out some administrative style letters as they are not appropriate
      imposters for the LILA dataset - see Dataset_Card.md.
    - Gets the text content from all of The Valley of the Shadow Augusta
      Virginia letters (link below) dataset and stores them as a DataFrame
      column.

    Returns a dataframe where rows represent letters, and columns are
    metadata fields, with the new `author` and `text_content` fields appended
    to the end.

    For more on The Valley of the Shadow dataset see Dataset_Card.md
    https://valley.newamericanhistory.org/search/letters/results?county=augusta

    :param data_dir: Full path to the directory containing the metadata and
    content subdirectories containing the raw data for all the letters.
    :type data_dir: string
    :rtype pandas.core.frame.DataFrame:
    """

    # Load the data
    df = pd.DataFrame(index=None)

    metadata_dir = os.path.join(data_dir, 'metadata')
    content_dir = os.path.join(data_dir, 'content')

    for file in os.listdir(metadata_dir):
        if file.endswith('.json'):
            row = None
            with open(os.path.join(metadata_dir, file)) as jd:
                data = json.load(jd)
                # Adapted from pbreach's answer on StackOverflow at
                # https://stackoverflow.com/a/21266043
                row = pd.json_normalize(data)
                row['base_file_name'] = file[:-5]

            content_file = os.path.join(content_dir, file[:-5] + '.html')

            with open(content_file) as hc:
                soup = BeautifulSoup(hc, "html.parser")
                # Adapted from code by Theo Vasilis
                # found at https://blog.apify.com/beautifulsoup-find-by-class/
                body = soup.find_all(class_='source__body')
                content = ''.join(
                    body[i].get_text()
                    for i
                    in range(len(body)))
                row['text_content'] = content

            df = pd.concat([df, row])

    # Create a copy of the title column to preserve original data
    df['author'] = df['title'].copy()

    # Adapted from Avinash Raj's answer on StackOverflow at
    # https://stackoverflow.com/a/30945796
    # 1. The author usually follows immediately after the substring 'from ' if it
    # exists in the title.
    # 2. Author's name usually sits immediately before the substring ' to ' if it
    # exists in the title.
    # 3. If both are true `1` should be followed for obtaining the authors name.
    #     - ...' from '...' to '.
    #     - ...' to '...' from '.
    #     - In both cases the author would follow the ' from '.

    # Remove everything before "from "
    df['author'] = df['author'].str.replace(r'^.*?[Ff]rom\s+', '', regex=True)

    # Remove everything after " to "
    df['author'] = df['author'].str.replace(r'\s+[Tt]o\s+.*$', '', regex=True)

    # Remove the commonly occuring substring "Freedmen's Bureau Records: "
    df['author'] = df['author'].str.replace("Freedmen's Bureau Records: ", '')

    # Remove the commonly ocurring substring "Freedman's Bureau Records: "
    df['author'] = df['author'].str.replace("Freedman's Bureau Records: ", '')

    # Remove the commonly occuring substring "Statement of "
    df['author'] = df['author'].str.replace("Statement of", '')

    # Remove the commonly occuring substring "Augusta County: "
    df['author'] = df['author'].str.replace("Augusta County: ", '')

    # Remove the commonly occuring substring "Augusta Country: "
    df['author'] = df['author'].str.replace("Augusta Country: ", '')

    # Remove the commonly occuring substring "Augusta: "
    df['author'] = df['author'].str.replace("Augusta: ", '')

    # Remove the commonly occuring substring "Franklin County: "
    df['author'] = df['author'].str.replace("Franklin County: ", '')

    # Remove the commonly occuring substring "Hanover County: "
    df['author'] = df['author'].str.replace("Hanover County: ", '')

    # Remove everything after January followed by a day
    df['author'] = df['author'].str.replace(r'\s+[Jj]anuary\s+\d+.*$', '', regex=True)

    # Remove everything after February followed by a day
    df['author'] = df['author'].str.replace(r'\s+[Ff]ebruary\s+\d+.*$', '', regex=True)

    # Remove everything after March followed by a day
    df['author'] = df['author'].str.replace(r'\s+[Mm]arch\s+\d+.*$', '', regex=True)

    # Remove everything after April followed by a day
    df['author'] = df['author'].str.replace(r'\s+[Aa]pril\s+\d+.*$', '', regex=True)

    # Remove everything after May followed by a day
    df['author'] = df['author'].str.replace(r'\s+[Mm]ay\s+\d+.*$', '', regex=True)

    # Remove everything after June followed by a day
    df['author'] = df['author'].str.replace(r'\s+[Jj]une\s+\d+.*$', '', regex=True)

    # Remove everything after July followed by a day
    df['author'] = df['author'].str.replace(r'\s+[Jj]uly\s+\d+.*$', '', regex=True)

    # Remove everything after August followed by a day
    df['author'] = df['author'].str.replace(r'\s+[Aa]ugust\s+\d+.*$', '', regex=True)

    # Remove everything after September followed by a day
    df['author'] = df['author'].str.replace(r'\s+[Ss]eptember\s+\d+.*$', '', regex=True)

    # Remove everything after October followed by a day
    df['author'] = df['author'].str.replace(r'\s+[Oo]ctober\s+\d+.*$', '', regex=True)

    # Remove everything after November followed by a day
    df['author'] = df['author'].str.replace(r'\s+[Nn]ovember\s+\d+.*$', '', regex=True)

    # Remove everything after December followed by a day
    df['author'] = df['author'].str.replace(r'\s+[Dd]ecember\s+\d+.*$', '', regex=True)

    # Clean up any remaining whitespace
    df['author'] = df['author'].str.strip()

    # Several items left are last will and testimates, receipts, or other
    # non-personal letters. There are also several marked as 'unknown'. These
    # sorts of entries need to be culled out.
    df = df[~df['author'].str.match(r'^Agreement ')]
    df = df[~df['author'].str.match(r'^Account Statement ')]
    df = df[~df['author'].str.match(r'^.*\s& ')]
    df = df[~df['author'].str.match(r'^.*\sand ')]
    df = df[~df['author'].str.match(r'^Circular ')]
    df = df[~df['author'].str.match(r'^Citizens of ')]
    df = df[~df['author'].str.match(r'^Contract of ')]
    df = df[~df['author'].str.match(r'^Deposition of ')]
    df = df[~df['author'].str.match(r'^.*\set\.? al\.?')]
    df = df[~df['author'].str.match(r"^Freedman's Bureau Records:")]
    df = df[~df['author'].str.match(r"^Invoices of ")]
    df = df[~df['author'].str.match(r"^List of Bureau Employees ")]
    df = df[~df['author'].str.match(r"^Loyalty Oath of ")]
    df = df[~df['author'].str.match(r"^Memo by ")]
    df = df[~df['author'].str.match(r"^Methodist Episcopal Church ")]
    df = df[~df['author'].str.match(r"^Petition")]
    df = df[~df['author'].str.match(r"^Receipt of ")]
    df = df[~df['author'].str.match(r"^Report of ")]
    df = df[~df['author'].str.match(r"^Special (Requisition )?Order ")]
    df = df[~df['author'].str.match(r"^T. W. Alexander, A. R. Wright, J. A. Stewart, T. S. Price, D. Scott, J. A. Johnson, Committee")]
    df = df[~df['author'].str.match(r"^Trustee Announcement for ")]
    df = df[~df['author'].str.match(r"^\[?[Uu]nkn?own")]
    df = df[~df['author'].str.match(r"^Various authors")]
    df = df[~df['author'].str.match(r"^Will of ")]

    """
    Some examples of remaining oddities:
     'Charles W. Baylor',
     'Charles W. Baylor [Bunk]',"

     'Frank W. [Lowes?]'

     'George S. Schreckhise',
     'George Schreckhise',

     'George Wils(?)',

     'John A McDonnell',
     'John A. McDonnell',
     'John A. Mcdonnell',

     'John P. Lightner',
     'John P. Lightner[?]',

     'L. (Letitia?) R. Smiley',
     ...
     'L. [Letitia?] R. Smiley',
     ...
     'Letitia R. Smiley'

     'Peter Hanger Jr.',
     'Peter Hanger, Jr.',

     'R. S. Lacey',
     'R.S. Lacey',

     'Roswell Waldo',
     'Roswell Waldo,',

     'Samuel F. Carson',
     'Samuel Franklin Carson',

     'Will A. Coulter',
     'Will Coulter',
    """

    # Replace everything between brackets with a space
    df['author'] = df['author'].str.replace(r'\[.*\]', ' ', regex=True)

    # Replace everything between parens with a space
    df['author'] = df['author'].str.replace(r'\(.*\)', ' ', regex=True)

    # Replace all special characters with spaces
    df['author'] = df['author'].str.replace(r'[^A-Za-z\s]', ' ', regex=True)

    # Lowercase
    df['author'] = df['author'].str.lower()

    # Remove middle names. This means we are assuming John B Good is the same as
    # John Good. In some cases this might not be true and some stylometric signal
    # could be muddied. This was deemed OK given the likely small effect it will
    # have and the efficiency gains this method can offer.
    df['author'] = df['author'].apply(remove_middle_name)

    # Cull any empty authors
    df = df[df['author'] != '']

    return df


data_path = os.path.join(ROOT_DIR, 'data')
lila_letters_data = os.path.join(data_path, 'original/LILA_imposters')

df = get_unique_author_strings_and_text_content(lila_letters_data)

# Get the path where the transformed data will be saved
target_dir = os.path.join(data_path, 'cleaning_in_process/imposters/tvots')

# Create the directory if it doesn't already exist
os.makedirs(target_dir, exist_ok=True)

# Loop through all `author`s in the dataframe and save the `text_content` to
# a .txt file with the name `base_file_name`.
for index, row in df.iterrows():
    author_dir = os.path.join(target_dir, row['author'])
    os.makedirs(author_dir, exist_ok=True)
    with open(
        f"{author_dir}/{row['base_file_name']}.txt", "w", encoding="utf-8"
    ) as f:
        f.write(row['text_content'])