# Max Pairs Calculation
Shorthand calculations determining the maximum number of class, genre, and imposter-type-genre balanced, same-genre pairs which could be generated given the training data. The `A` class's 'Pros/Fiction' sub-split was found to be the limiting factor resulting in a total number of possible pairs of 20,720.

```txt
A
GENRE: Pros/Fiction
files_in_genre:
    ebil_eye.txt
    jinny.txt
    kunnoo_sperits_others.txt
    yule_log.txt
words_in_genre: 17278
chunks_in_genre: 36
---------------------------------------
GENRE: Pros/Historical & Biographical
files_in_genre:
    across_my_path.txt
    bugles_of_gettysburg.txt
    fourth_massachusetts_cavalry.txt
    literary_hearthstones_dixie.txt
    pickett_his_men.txt
    what_happened_to_me.txt
words_in_genre: 261639
chunks_in_genre: 556
---------------------------------------

Ratios:
    Pros/Fiction: 36/(36 + 556)                    = 0.060810810810810814
    Pros/Historical & Biographical: 556/(36 + 556) = 0.9391891891891891

The greatest number of intra-genre same author pairs by genre that can be
created are:

(36 * (36 - 1))/2   =     630
(556 * (556 - 1))/2 = 154,290

Therefore, the greatest number of total pairs that can be created is:

630 / x = 0.060810810810810814 / 0.9391891891891891
630 = (0.060810810810810814 / 0.9391891891891891) * x
630 / (0.060810810810810814 / 0.9391891891891891) = x = 9,730 < 154,290

x / 154290 = 0.060810810810810814 / 0.9391891891891891
x = (0.060810810810810814 / 0.9391891891891891) * 154290
9990 = (0.060810810810810814 / 0.9391891891891891) * 154290 > 630

So Pros/Fiction chunks is the limiting factor and the most total pairs is:

630 + 9730 = 10,360

We have two imposter types, which must share their half of the total model
pairs equally, so each imposter gets:

(10360/2)/2 = 2,590 chunks

The number of chunks per imposter type, per genre are:

Pros/Fiction                   = 2590 * 0.060810810810810814 =   157.5
Pros/Historical & Biographical = 2590 * 0.9391891891891891   = 2,432.5

notA
IMPOSTER TYPE: George
    GENRE: Pros/Fiction
    files_in_genre_by_impt:
        mohun_or_the_last_days_of_lee_and_his_paladins.txt
    words_in_genre_by_impt: 163689
    chunks_in_genre_by_impt: 353
    ---------------------------------------
    GENRE: Pros/Historical & Biographical
    files_in_genre_by_impt:
        from_manassas_to_appomattox.txt
        heritage_of_the_south.txt
        recollections_and_letters.txt
        the_rise_and_fall_of_the_confederate_government_volume_1.txt
    words_in_genre_by_impt: 464248
    chunks_in_genre_by_impt: 977
    ---------------------------------------

    The greatest number of George-type chunks that can be created are:

    Pros/Fiction                   =  36 * 353 =  12,708 >   157.5
    Pros/Historical & Biographical = 556 * 977 = 543,212 > 2,431.5

IMPOSTER TYPE: LaSalle
    GENRE: Pros/Fiction
    files_in_genre_by_impt:
        balcony_stories.txt
    words_in_genre_by_impt: 34796
    chunks_in_genre_by_impt: 76
    ---------------------------------------
    GENRE: Pros/Historical & Biographical
    files_in_genre_by_impt:
        a_confederate_girls_diary.txt
        belle_boyd_in_camp_and_prison_vol_1.txt
        belle_boyd_in_camp_and_prison_vol_2.txt
        diary_from_dixie.txt
        lee_and_longstreet.txt
        my_imprisonment_and_the_first_year_of_abolition_rule_at_washington.txt
        recollections_grave_and_gay.txt
    words_in_genre_by_impt: 552533
    chunks_in_genre_by_impt: 1178
    ---------------------------------------

    The greatest number of LaSalle-type chunks that can be created are:

    Pros/Fiction                   =  36 *   76 =   2,736 >   157.5
    Pros/Historical & Biographical = 556 * 1178 = 654,968 > 2,431.5

Therefore the same-author pairs imposes the limit on how many pairs can be requested, which is:

10,360 * 2 = 20,720
```