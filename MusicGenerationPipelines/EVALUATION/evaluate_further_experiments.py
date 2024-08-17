import pandas as pd
import os
from EVALUATION.compare_text_and_audio_features import evaluate_similarity
from ebooklib import epub
import json
from Pipeline_MusicLDM import download_book, read_book, is_valid, chapter_to_str



def get_chapter(book_name, chapter_ind):
    try:
        with open("all_book_chapters_mod.json") as f:
            books = json.load(f)
        chap = books[book_name][chapter_ind-1]
        return chap
    except:
        return None
    


def evaluate_further_experiments(path, output_file, caption_text_flag, eval_func, paragraph_limit=None, chapter_char_limit=None):

    # Read in the CSV as a DataFrame
    df = pd.read_csv('EVAL_FURTHER/eval_further_experiments.csv', index_col=0)

    # Define the relative path to the "music_from_AudioLDM" folder
    relative_path = path

    # Get the absolute path based on the current working directory
    absolute_path = os.path.abspath(relative_path)

    # Iterate through the folder and print each file name
    for folder in os.listdir(absolute_path):
        if folder[0] != ".":
            row = folder
            print(row)
            book_name = folder + ".epub"
            if row not in df.index:
                df.loc[row] = [None] * len(df.columns)
            for subfolder in os.listdir(absolute_path + "/" + folder):
                if subfolder[0] != ".":
                    column = subfolder
                    print("\t", subfolder)
                    if column not in df.columns:
                        df[column] = None
                    with open(absolute_path + "/" + folder + "/" + subfolder + "/prompt.txt", encoding="utf-8") as f:
                        prompt_text = f.read()
                        prompt_text = prompt_text[:chapter_char_limit]
                    for file in os.listdir(absolute_path + "/" + folder + "/" + subfolder):
                        if file.endswith(".wav"):
                            print("\t\t", file)
                            file_components = file.split("_")
                            # get the component after "chapter"
                            index = file_components.index("chapter") + 1
                            chapter_ind = int(file_components[index])

                            print("\t\t", book_name)
                            print("\t\t", chapter_ind)
                            chapter_text = get_chapter(book_name, chapter_ind)
                            if not chapter_text:
                                print("hi")
                                continue

                            if paragraph_limit == 1:
                                chapter_text = chapter_text
                                if len(chapter_text) > 1000:
                                    chapter_text = chapter_text[:1000]
                            elif paragraph_limit > 1:
                                raise ValueError("Paragraph limit must be 1 or None")

                            if caption_text_flag:
                                text = prompt_text
                            else:
                                text = chapter_text
                            comparison = eval_func(text, absolute_path + "/" + folder + "/" + subfolder + "/" + file)
                            print("\t\t", comparison)
                            df.loc[row, column] = comparison[0]

    # Save to CSV
    df = df.round(2)
    df.to_csv(f'{output_file}.csv', index=True)
