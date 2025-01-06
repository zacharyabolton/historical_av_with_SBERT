---
cssclasses:
  - word-break-all-third-col
---
This is a dataset for Authorship Verification concerning the love letters attributed to General George E. Pickett C.S.A., published posthumously by his wife, LaSalle Corbell Pickett in her books _The Heart of a Soldier, As Revealed in the Intimate Letters of Gen'l George E. Pickett, C.S.A. (1913)_ and _Soldier of the South: General Pickett's War Letters to His Wife (edited by Arthur Crew Inman, 1928)_.

It contains the love letters, known works of LaSalle, and 'imposter' works - works chosen for their authors' socio-demographic similarity to LaSalle or the general.
## Structure
The contents of this dataset are structured in a hierarchical directory structure shown below:
```txt
.
├── cleaned
│   ├── A
│   │   ├── across_my_path.txt
│   │   └── ...
│   ├── excluded
│   │   ├── digging_through_to_manila.txt
│   │   └── ...
│   ├── notA
│   │   ├── a_confederate_girls_diary.txt
│   │   └── ...
│   └── U
│       ├── heart_of_soldier.txt
│       └── ...
├── cleaning_in_process
│   ├── george_pickett
│   │   └── misc_excerpts.txt
│   ├── imposters
│   │   ├── a_confederate_girls_diary/
│   │   └── ...
│   ├── lasalle_corbell_pickett/
│   │   ├── across_my_path/
│   │   └── ...
│   └── love_letters
│       ├── heart_of_soldier/
│       └── soldier_of_the_south/
├── normalized
│   ├── DV-MA-k-300
│   │   ├── A
│   │   ├── notA
│   │   └── U
│   ├── DV-...
│   ├── DV-SA-k-300
│   │   └── ...
│   ├── DV-...
│   └── undistorted
│       └── ...
├── original
│   ├── lasalle_corbell_pickett
│   │   ├── acrossmypath00pick_djvu.xml
│   │   └── ...
│   ├── LIL_imposters
│   │   ├── balcony_stories.html
│   │   └── ...
│   ├── LILA_imposters
│   │   ├── a_confederate_girls_diary.html
│   │   └── ...
│   └── love_letters
│       ├── heartofsoldieras00pick_djvu.xml
│       └── soldier_of_the_south
└── test
    └── ...
```

Where:
- `original/` are the original works.
- `cleaned/` are the original works with extraneous data removed, such as OCR artifacts, markup, metadata, and non-authorial quotations and insertions.
- `normalized/` is the cleaned data that has been lowercased and had non alphanumeric characters removed.
    - `normalized/` also contains its own distorted views based on the procedure describe in Stamatatos et al. (2017) [^1], where the first part of the subdirectories name indicates the algorithm used (DV-MA/DV-SA) and the second part indicates the $k$ value used (k-300, k-3000, etc...).
    - `normalized/` is intended for model ingestion.
- `cleaning_in_process/` are intermediate versions of the works before `cleaned/`. 
- `test/` is fake data for debugging and testing.
    - `test/` should be empty as the data is created and cleaned up programmatically by the `pytest` test suite.

The `A`, `notA`, and `U` datasets are especially named for model ingestion as the model expects a path to the parent of these three directories. `A` indicates works by the known author. `notA` indicates 'imposter' works. `U` is the work(s) of unknown authorship.
## ORIGINAL LIL DATASET CONTENTS
**<mark>A NOTE ON THE CONTENTS THIS OF DATASET:</mark>** Given that these works are written by Southern confederates during the time of the American Civil War, the contents are highly objectionable, consisting of xenophobic and anti-black rhetoric, pro-slavery arguments, and eye-dialect, among other problematic speech. **Please inspect this data with caution**.
##### LaSalle Corbell Pickett
**COPYWRITE STATUS**
Since all works were published before 1923, they are in the public domain: <https://guides.library.oregonstate.edu/copyright/publicdomain#>

| **Item**                                                                                                                                   | **Source**          | **URL**                                                                                                            | **Acquired Date** |
|--------------------------------------------------------------------------------------------------------------------------------------------|---------------------|--------------------------------------------------------------------------------------------------------------------|-------------------|
| **Pickett and His Men (1899)**                                                                                                             | Library of Congress | https://tile.loc.gov/storage-services/public/gdcmassbookdig/picketthismen01pick/picketthismen01pick_djvu.xml       | 2024-05-19        |
| **Kunnoo Sperits and Others (1900)**                                                                                                       | Internet Archive    | https://archive.org/download/kunnoosperitsoth00pickiala/kunnoosperitsoth00pickiala_djvu.xml                        | 2024-05-20        |
| **Yule Log (1900)**                                                                                                                        | HathiTrust.org      | https://babel.hathitrust.org/cgi/pt?id=hvd.hn6gb3                                                                  | 2024-06-17        |
| **Ebil Eye (1901)**                                                                                                                        | HathiTrust.org      | https://babel.hathitrust.org/cgi/pt?id=osu.32435058048737                                                          | 2024-05-22        |
| **Jinny (1901)**                                                                                                                           | HathiTrust.org      | https://babel.hathitrust.org/cgi/pt?id=osu.32435079799953                                                          | 2024-05-22        |
| **Digging Through to Manila (1905)**                                                                                                       | Library of Congress | https://tile.loc.gov/storage-services/public/gdcmassbookdig/diggingthroughto00pick/diggingthroughto00pick_djvu.xml | 2024-05-22        |
| **Literary Hearthstones of Dixie (1912)**                                                                                                  | Project Gutenberg   | https://www.gutenberg.org/ebooks/16622                                                                             | 2024-05-22        |
| **The Bugles of Gettysburg (1913)**                                                                                                        | Internet Archive    | https://ia801304.us.archive.org/33/items/buglesofgettysbu0000lasa/buglesofgettysbu0000lasa_djvu.xml                | 2024-05-22        |
| **Across My Path: Memories of People I Have Known (1916)**                                                                                 | Library of Congress | https://tile.loc.gov/storage-services/public/gdcmassbookdig/acrossmypath00pick/acrossmypath00pick_djvu.xml         | 2024-05-22        |
| **What Happened to Me … (1917)**                                                                                                           | Project Gutenberg   | https://www.gutenberg.org/ebooks/50001                                                                             | 2024-06-24        |
| **The Fourth Massachusetts Cavalry in the Closing Scenes of the War for the Maintenance of the Union, from Richmond to Appomattox (1910)** | Project Gutenberg   | https://www.gutenberg.org/ebooks/31977                                                                             | 2024-05-22        |
##### Love Letters
**COPYWRITE STATUS**
_The Heart of a Soldier, As Revealed in the Intimate Letters of Gen'l George E. Pickett, C.S.A._ was published before 1923, and therefore is in the public domain: <https://guides.library.oregonstate.edu/copyright/publicdomain#>
_Soldier of the South: General Pickett's War Letters to His Wife_ is stated as belonging to the public domain by the [HathiTrust Digital Library](https://www.hathitrust.org/the-collection/).

| **Item**                                                                                                  | **Source**          | **URL**                                                                                                            | **Acquired Date** |
|-----------------------------------------------------------------------------------------------------------|---------------------|--------------------------------------------------------------------------------------------------------------------|-------------------|
| **The Heart of a Soldier, As Revealed in the Intimate Letters of Gen'l George E. Pickett, C.S.A. (1913)** | Library of Congress | https://tile.loc.gov/storage-services/public/gdcmassbookdig/heartofsoldieras00pick/heartofsoldieras00pick_djvu.xml | 2024-05-22        |
| **Soldier of the South: General Pickett's War Letters to His Wife (edited by Arthur Crew Inman, 1928)**   | HathiTrust.org      | https://babel.hathitrust.org/cgi/pt?id=uc1.$b61025                                                                 | 2024-05-22        |
##### Imposters
**COPYWRITE STATUS**
All works are stated as belonging to the public domain by [Project Gutenberg](https://www.gutenberg.org/).

| **Item**                                                                                                           | **Source**        | **URL**                                | **Acquired Date** |
|--------------------------------------------------------------------------------------------------------------------|-------------------|----------------------------------------|-------------------|
| **A Diary from Dixie by Mary Boykin Chesnut (1905)**                                                               | Project Gutenberg | https://www.gutenberg.org/ebooks/60908 | 2024-06-23        |
| **Vashti; Or, Until Death Us Do Part by Augusta J. Evans (1869)**                                                  | Project Gutenberg | https://www.gutenberg.org/ebooks/31620 | 2024-06-23        |
| **A Speckled Bird by Augusta J. Evans (1902)**                                                                     | Project Gutenberg | https://www.gutenberg.org/ebooks/36029 | 2024-06-23        |
| **Macaria by Augusta J. Evans (1863)**                                                                             | Project Gutenberg | https://www.gutenberg.org/ebooks/27811 | 2024-06-23        |
| **St. Elmo by Augusta J. Evans (1866)**                                                                            | Project Gutenberg | https://www.gutenberg.org/ebooks/4553  | 2024-06-23        |
| **Balcony Stories by Grace Elizabeth King (1893)**                                                                 | Project Gutenberg | https://www.gutenberg.org/ebooks/11514 | 2024-06-23        |
| **Recollections and Letters of General Robert E. Lee by Robert E. Lee (1904)**                                     | Project Gutenberg | https://www.gutenberg.org/ebooks/2323  | 2024-06-23        |
| **From Manassas to Appomattox: Memoirs of the Civil War in America by James Longstreet (1896)**                    | Project Gutenberg | https://www.gutenberg.org/ebooks/38418 | 2024-06-23        |
| **Lee and Longstreet at High Tide: Gettysburg in the Light of the Official Records by Helen D. Longstreet (1904)** | Project Gutenberg | https://www.gutenberg.org/ebooks/44459 | 2024-06-23        |
| **The Heritage of The South by Jubal Anderson Early (1867)**                                                       | Project Gutenberg | https://www.gutenberg.org/ebooks/63254 | 2024-06-23        |
## ADDED LILA DATA
**COPYWRITE STATUS**

**Misc. Letters from various Confederates living in Augusta County Virginia during, and immediately before and after the U.S. Civil War.** (_TVOTS_)
    Distributed under a [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/)
    Distributed by [**The Valley of the Shadow Project**](https://valley.newamericanhistory.org/):
    	Franklin County: All letters (1850-1880), Valley of the Shadow: Two Communities in the American Civil War (https://valley.newamericanhistory.org/search/letters/results?county=augusta).
    These works were preprocessed and obfuscated for ingestion into a machine learning algorithm focused on the task of authorship verification.
    This data and any data, models, or intellectual product derived thereof are not for commercial use.

_All other workds_ were published before 1923, and therefore is in the public domain: <https://guides.library.oregonstate.edu/copyright/publicdomain#>

| **Item**                                                                                                                                      | **Source**                       | **URL**                                             | **Acquired Date** |
| --------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | --------------------------------------------------- | ----------------- |
| **Misc. Letters from various Confederates living in Augusta County Virginia during, and immediately before and after the U.S. Civil War.** \* | The Valley of the Shadow Project | https://valley.newamericanhistory.org/              | 2024-12-18        |
| **A Life of Gen. Robert E. Lee by John Esten Cooke (1871)**                                                                                   | Project Gutenberg                | https://www.gutenberg.org/ebooks/10692              | 2024-12-26        |
| **Mohun; Or, the Last Days of Lee and His Paladins. by John Esten Cooke (1869)**                                                              | Project Gutenberg                | https://www.gutenberg.org/ebooks/8424               | 2024-12-26        |
| **The Rise and Fall of the Confederate Government, Volume 1 by Jefferson Davis (1881)**                                                       | Project Gutenberg                | https://www.gutenberg.org/ebooks/19831              | 2024-12-27        |
| **Belle Boyd in Camp and Prison. In Two Volumes. Vol. I. by Belle Boyd (1865)**                                                               | Documenting the American South   | https://docsouth.unc.edu/fpn/boyd1/menu.html        | 2024-12-27        |
| **Belle Boyd in Camp and Prison. In Two Volumes. Vol. II. by Belle Boyd (1865)**                                                              | Documenting the American South   | https://docsouth.unc.edu/fpn/boyd2/menu.html        | 2024-12-27        |
| **My Imprisonment and the First Year of Abolition Rule at Washington by Rose O'Neal Greenhow (1853)**                                         | Documenting the American South   | https://docsouth.unc.edu/fpn/greenhow/menu.html     | 2024-12-27        |
| **Recollections Grave and Gay by Constance Cary Harrison (1911)**                                                                             | Documenting the American South   | https://docsouth.unc.edu/fpn/harrison/harrison.html | 2024-12-27        |
| **A Confederate Girl's Diary by Sarah Morgan Dawson (1913)**                                                                                  | Documenting the American South   | https://docsouth.unc.edu/fpn/dawson/dawson.html     | 2024-12-27        |

\* _NOTE_: TVOTS texts only appear in the `/original` subdirectory, as it was decided not to include them in the final project, due to concerns with data biasing. So, preprocessing was not performed in this instance.

---

[^1]: Efstathios Stamatatos. 2017. Authorship Attribution Using Text Distortion. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers, Association for Computational Linguistics, Valencia, Spain, 1138–1149. Retrieved from [https://aclanthology.org/E17-1107](https://aclanthology.org/E17-1107)