import json
import sys
import os
import pandas as pd
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


def get_unique_author_strings(metadata_dir):
    """
    Augments The Valley of the Shadow Augusta Virginia letters (link below)
    dataset with a field for the author as represented by a transformed
    snake_case string. Culls out some administrative style letters as they
    are not appropriate imposters for the LILA dataset - see Dataset_Card.md.
    For more on The Valley of the Shadow dataset see Dataset_Card.md
    Returns a dataframe where rows represent letters, and columns are
    metadata fields, with the new `author` field appended to the end.

    https://valley.newamericanhistory.org/search/letters/results?county=augusta

    :param metadata_dir: full path to the directory containing the metadata
    JSON files for each letter.
    :type metadata_dir: string
    :rtype pandas.core.frame.DataFrame:
    """
    data_path = os.path.join(ROOT_DIR, 'data')
    lila_letters_metadata = os.path.join(
        data_path,
        'original/LILA_imposters/metadata')

    # Load the data
    df = pd.DataFrame(index=None)
    for file in os.listdir(lila_letters_metadata):
        if file.endswith('.json'):
            with open(os.path.join(lila_letters_metadata, file), 'r') as jd:
                data = json.load(jd)
                # Adapted from pbreach's answer on StackOverflow at
                # https://stackoverflow.com/a/21266043
                row = pd.json_normalize(data)
                row['local_file'] = file
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