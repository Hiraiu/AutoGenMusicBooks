import pandas as pd
import os
from EVALUATION.compare_text_and_audio_features import evaluate_similarity
import json

def evaluate_initial_experiments(output_file, caption_text_flag, musicgenmodel, eval_func, chapter_char_limit=None):

    with open("INITIAL_EXPERIMENTS/all_book_chapters.json") as f:
        all_chapters = json.load(f)

    #read in the csv as a dataframe
    df = pd.read_csv('EVAL_INITIAL/eval_initial_experiments.csv')
    print(df)

    if musicgenmodel == "MusicGen":
        model = "music_from_MusicGenForTori"
    elif musicgenmodel == "AudioLDM":
        model = "music_from_AudioLDM"
    elif musicgenmodel == "MusicLDM":
        model = "music_from_MusicLDM"

    # Define the relative path to the "music_from_AudioLDM" folder
    relative_path = f"INITIAL_EXPERIMENTS/ebooks/{model}"

    # Get the absolute path based on the current working directory
    absolute_path = os.path.abspath(relative_path)

    # Iterate through the folder and print each file name
    for folder in os.listdir(absolute_path):
        if folder.startswith("."):
            continue
        
        row = folder
        row = row.replace("_", "-")
        print(row)

        for subfolder in os.listdir(absolute_path + "/" + folder):
            if subfolder.startswith("."):
                continue
            if subfolder == "intro":
                for subsubfolder in os.listdir(absolute_path + "/" + folder + "/" + subfolder):
                    if not subsubfolder.startswith("."):
                        column_name = subfolder + "-" + subsubfolder
                        column_name = column_name.replace("_", "-")
                        print("\t", column_name)
                        for sssf in os.listdir(absolute_path + "/" + folder + "/" + subfolder + "/" + subsubfolder):
                            if not sssf.startswith("."):
                                results_ls = []
                                for chapter in os.listdir(absolute_path + "/" + folder + "/" + subfolder + "/" + subsubfolder + "/" + sssf):
                                    if not chapter.startswith("."):
                                        # extract the text from prompt.txt
                                        with open(absolute_path + "/" + folder + "/" + subfolder + "/" + subsubfolder + "/" + sssf + "/" + chapter + "/prompt.txt", encoding="utf-8") as f:
                                            text = f.read()
                                        for file in os.listdir(absolute_path + "/" + folder + "/" + subfolder + "/" + subsubfolder + "/" + sssf + "/" + chapter):
                                            if file.endswith(".wav"):
                                                print("\t\t", file)
                                                # use regex to look at the chapter number in the file
                                                chapter_nr = file.split("_")[4]
                                                if caption_text_flag is False:
                                                    text = all_chapters["the_three_taps.epub"][int(chapter_nr)-1]
                                                text = text[:chapter_char_limit] if chapter_char_limit is not None else text
                                                
                                                comparison = eval_func(text, absolute_path + "/" + folder + "/" + subfolder + "/" + subsubfolder + "/" + sssf + "/" + chapter + "/" + file)
                                                print("\t\t", comparison)
                                                results_ls.append(comparison[0])
                                        # get the average of the results_ls
                                        average = sum(results_ls) / len(results_ls)
                                        df.loc[row, column_name] = average


            elif subfolder == "ollama":
                for subsubfolder in os.listdir(absolute_path + "/" + folder + "/" + subfolder):
                    if not subsubfolder.startswith("."):
                        for sssf in os.listdir(absolute_path + "/" + folder + "/" + subfolder + "/" + subsubfolder):
                            if not sssf.startswith("."):
                                if sssf != "the_three_taps":
                                    column_name = subfolder + "-" + subsubfolder + "-" + sssf
                                    column_name = column_name.replace("_", "-")
                                    print("\t", column_name)
                                    for ssssf in os.listdir(absolute_path + "/" + folder + "/" + subfolder + "/" + subsubfolder + "/" + sssf):
                                        results_ls = []
                                        for chapter in os.listdir(absolute_path + "/" + folder + "/" + subfolder + "/" + subsubfolder + "/" + sssf + "/" + ssssf):
                                                                    # extract the text from prompt.txt
                                            with open(absolute_path + "/" + folder + "/" + subfolder + "/" + subsubfolder + "/" + sssf + "/" + ssssf + "/" + chapter + "/prompt.txt", encoding="utf-8") as f:
                                                text = f.read()
                                            for file in os.listdir(absolute_path + "/" + folder + "/" + subfolder + "/" + subsubfolder + "/" + sssf + "/" + ssssf + "/" + chapter):
                                                if file.endswith(".wav"):
                                                    print("\t\t", file)
                                                    chapter_nr = file.split("_")[4]
                                                    if caption_text_flag is False:
                                                        text = all_chapters["the_three_taps.epub"][int(chapter_nr)-1]
                                                    text = text[:chapter_char_limit] if chapter_char_limit is not None else text
                                                    comparison = eval_func(text, absolute_path + "/" + folder + "/" + subfolder + "/" + subsubfolder + "/" + sssf + "/" + ssssf + "/" + chapter + "/" + file)
                                                    print("\t\t", comparison)
                                                    results_ls.append(comparison[0])
                                        average = sum(results_ls) / len(results_ls)
                                        df.loc[row, column_name] = average
                                else:
                                    column_name = subfolder + "-" + subsubfolder
                                    column_name = column_name.replace("_", "-")
                                    print("\t", column_name)
                                    results_ls = []
                                    for chapter in os.listdir(absolute_path + "/" + folder + "/" + subfolder + "/" + subsubfolder + "/" + sssf):
                                                                # extract the text from prompt.txt
                                        with open(absolute_path + "/" + folder + "/" + subfolder + "/" + subsubfolder + "/" + sssf + "/" + chapter + "/prompt.txt", encoding="utf-8") as f:
                                            text = f.read()
                                        for file in os.listdir(absolute_path + "/" + folder + "/" + subfolder + "/" + subsubfolder + "/" + sssf + "/" + chapter):
                                            if file.endswith(".wav"):
                                                print("\t\t", file)
                                                chapter_nr = file.split("_")[4]
                                                if caption_text_flag is False:
                                                    text = all_chapters["the_three_taps.epub"][int(chapter_nr)-1]
                                                text = text[:chapter_char_limit] if chapter_char_limit is not None else text
                                                comparison = eval_func(text, absolute_path + "/" + folder + "/" + subfolder + "/" + subsubfolder + "/" + sssf + "/" + chapter + "/" + file)
                                                print("\t\t", comparison)
                                                results_ls.append(comparison[0])
                                                        # get the average of the results_ls
                                    average = sum(results_ls) / len(results_ls)
                                    df.loc[row, column_name] = average
            


            elif subfolder == "other":
                column_name = "other"
                print("\t", column_name)
                for b in os.listdir(absolute_path + "/" + folder + "/" + subfolder):
                    if not b.startswith("."):
                        results_ls = []
                        for chapter in os.listdir(absolute_path + "/" + folder + "/" + subfolder + "/" + b):
                                                    # extract the text from prompt.txt
                            with open(absolute_path + "/" + folder + "/" + subfolder + "/" + b + "/" + chapter + "/prompt.txt", encoding="utf-8") as f:
                                text = f.read()
                            for file in os.listdir(absolute_path + "/" + folder + "/" + subfolder + "/" + b + "/" + chapter):
                                if file.endswith(".wav"):
                                    print("\t\t", file)
                                    chapter_nr = file.split("_")[4]
                                    if caption_text_flag is False:
                                        text = all_chapters["the_three_taps.epub"][int(chapter_nr)-1]
                                    text = text[:chapter_char_limit] if chapter_char_limit is not None else text
                                    comparison = eval_func(text, absolute_path + "/" + folder + "/" + subfolder + "/" + b + "/" + chapter + "/" + file)
                                    print("\t\t", comparison)
                                    results_ls.append(comparison[0])
                        average = sum(results_ls) / len(results_ls)
                        df.loc[row, column_name] = average



    print(df)
    # Save the updated dataframe to a new CSV file, keep the row names
    # round all values to 2 decimal places
    df = df.round(2)
    df.to_csv(f'{output_file}.csv', index=True)
            

