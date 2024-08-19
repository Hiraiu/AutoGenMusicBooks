import os
import sys
import nltk
import urllib.request
import scipy
import torch
import subprocess
import json
from OLlama import interact
from ebooklib import epub
from bs4 import BeautifulSoup
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from diffusers import MusicLDMPipeline
from collections import Counter
from extract_ADJ_NOUN_pairs import get_most_common_adj_noun_pairs

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def download_book(url, file_name):
    """
    Downloads a book from a given URL and saves it to a specified file name.
    """
    with urllib.request.urlopen(url) as response:
        s = response.read()
    with open(file_name, "wb") as f:
        f.write(s)
    return file_name

def read_book(file_name):
    """
    Reads an EPUB book from a given file name.
    """
    return epub.read_epub(file_name, {"ignore_ncx": True})

def chapter_to_str(chapter):
    """
    Converts a chapter object into a string.
    """
    soup = BeautifulSoup(chapter.get_body_content(), 'html.parser')
    text = [para.get_text() for para in soup.find_all('p')]
    return ' '.join(text)

def is_valid(epub_book):
    """
    Validates the structure of an epub book. Checks if the table of content, the spine of the book and stylesheets are present. Ensures that the chapters are not empty.
    """
    # If table of contents is missing
    if not epub_book.toc:
        print('Table of content is missing')
        return False
    # If the spine is missing
    if not epub_book.spine:
        print('Spine is missing')
        return False

    # If chapters are empty
    for item in epub_book.spine:
        if isinstance(item, epub.EpubHtml):
            if not item.content.strip():
                print(f"Invalid book: Chapter '{item.get_id()}' is empty.")
                return False
    return True


def start_musicgen(gpu=True):
    """Load musicgen model for text-to-music generation
    Source: https://huggingface.co/facebook/musicgen-small"""
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    return model

def generate_music(model, prompt, music_path, duration_seconds):
    """Generate a music snippet based on a text prompt"""
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt")
    
    # Estimate the number of tokens needed for the desired duration
    # Assuming each token corresponds to a small fraction of a second (adjust as necessary)
    tokens_per_second = 50 
    max_new_toks = tokens_per_second * duration_seconds
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=max_new_toks)
    sampling_rate = model.config.audio_encoder.sampling_rate

    scipy.io.wavfile.write(music_path, rate=sampling_rate, data=audio_values[0, 0].numpy())
    print(f"Music snippet saved to {music_path}")


def create_bg_music(paragraph, book_id, chapter_id, music_dir, generation_params, music_gen=False):
    """Generate background music using the first paragraph of a chapter as prompt"""
    # add subfolders to music_dir according to the specifications
        
    prompt_intro = generation_params["prompt_intro"]
    
    if generation_params["hardcoded_prompt"]:
        prompt = generation_params["hardcoded_prompt"]
        folder = prompt[:25].replace(" ", "_")
        specification = prompt[:25].replace(" ", "_")
        music_dir = os.path.join(music_dir, "hardcoded_prompts")
        music_dir = os.path.join(music_dir, folder)
        os.makedirs(music_dir, exist_ok=True)
    elif generation_params["adj_flag"] or generation_params["verb_flag"] or generation_params["noun_flag"]:
        POS = "JJ" if generation_params["adj_flag"] else "VB" if generation_params["verb_flag"] else "NN"
        prompt = get_n_most_frequent(paragraph, generation_params["most_frequent_n_words"], POS)
        specification = prompt.replace(' ', '_')
        # create if not exists and add a subfolder for the adjectives
        folder = 'adjectives' if generation_params["adj_flag"] else 'verbs' if generation_params["verb_flag"] else 'nouns'
        music_dir = os.path.join(music_dir, folder)
        os.makedirs(music_dir, exist_ok=True)
    elif generation_params["adjnoun_flag"]:
        prompt = get_most_common_adj_noun_pairs(paragraph, generation_params["most_frequent_n_words"])
        specification = prompt.replace(' ', '_')
        # create if not exists and add a subfolder for the adjectives
        folder = "adj_noun_pairs"
        music_dir = os.path.join(music_dir, folder)
        os.makedirs(music_dir, exist_ok=True)
    elif generation_params["first_n_paragraphs"] is not None:
        music_dir = os.path.join(music_dir, 'first_n_paragraphs')
        os.makedirs(music_dir, exist_ok=True)
        music_dir = os.path.join(music_dir, str(generation_params["first_n_paragraphs"]) + '_paragraphs')
        os.makedirs(music_dir, exist_ok=True)
        # the prompt is the paragraph until the first \n
        prompt = paragraph
        prompted_paragraph = prompt
        specification = 'first_' + str(generation_params["first_n_paragraphs"]) + '_paragraphs'
    else:
        music_dir = os.path.join(music_dir, 'first_n_chars')
        os.makedirs(music_dir, exist_ok=True)
        music_dir = os.path.join(music_dir, str(generation_params["first_n_chars"]) + '_chars')
        os.makedirs(music_dir, exist_ok=True)
        prompt = paragraph[:generation_params["first_n_chars"]]
        specification = 'first_' + str(generation_params["first_n_chars"]) + '_chars'

    if not generation_params["hardcoded_prompt"]:
        if generation_params["ollama_prompt"]:
            if generation_params["intro_in_ollama"]:
                prompt = prompt_intro + prompt
            music_dir = os.path.join(music_dir, 'ollama')
            os.makedirs(music_dir, exist_ok=True)
            ollama_prompt = generation_params["ollama_prompt"]
            print()
            print("Ollama is prompted with: ")
            print(ollama_prompt + prompt)
            print()
            prompt = interact(ollama_prompt + prompt)

            if "\n\n" in prompt:
                prompt = prompt.split('\n\n')[1]
            
            specification = 'ollama_' + specification
        
        if prompt_intro:
            # create if not exists and add a subfolder for the adjectives
            if not generation_params["intro_in_ollama"]:
                music_dir = os.path.join(music_dir, 'intro')
                os.makedirs(music_dir, exist_ok=True)
                prompt = prompt_intro + prompt
            else:
                music_dir = os.path.join(music_dir, 'intro_in_ollama_prompt')
                os.makedirs(music_dir, exist_ok=True)
            specification = 'intro_' + specification
        
        prompt_to_be_looked_at = prompt_intro if generation_params["intro_in_ollama"] else prompt
        if "piano" in prompt_to_be_looked_at:
            music_dir = os.path.join(music_dir, "piano")
            os.makedirs(music_dir, exist_ok=True)
        elif "instrumental" in prompt_to_be_looked_at:
            music_dir = os.path.join(music_dir, "instrumental")
            os.makedirs(music_dir, exist_ok=True)
        else:
            music_dir = os.path.join(music_dir, "other")
            os.makedirs(music_dir, exist_ok=True)
    
    music_dir = os.path.join(music_dir, book_id)
    os.makedirs(music_dir, exist_ok=True)

    music_dir = os.path.join(music_dir, chapter_id)
    os.makedirs(music_dir, exist_ok=True)


    prompt = prompt.strip().replace('\n', ' ').replace('\\"', '')

    if music_gen is not False:
        # make a txt file with the prompt and save it to the music_dir
        with open(os.path.join(music_dir, 'prompt.txt'), 'w') as f:
            f.write(prompt)
        with open(os.path.join(music_dir, 'prompted_paragraph.txt'), 'w') as f:
            f.write(prompted_paragraph)
    
    ldm_dir = music_dir.replace("music_from_MusicGenForTori", "music_from_MusicLDM")
    os.makedirs(ldm_dir, exist_ok=True)
    with open(os.path.join(ldm_dir, 'prompt.txt'), 'w') as f:
        f.write(prompt)
    with open(os.path.join(ldm_dir, 'prompted_paragraph.txt'), 'w') as f:
        f.write(prompted_paragraph)

    text_mus_id = book_id + '_' + chapter_id + '_bgmus_' + specification

    music_path = os.path.join(music_dir, f'{text_mus_id}.wav')
    ldm_path = os.path.join(ldm_dir, f'{text_mus_id}.wav')

    audio_length_in_s = int(generation_params["audio_length_in_s"])

    
    # Generate music if it doesn't exist
    if not os.path.isfile(music_path):
        print()
        print(f"Creating new music snippet for {text_mus_id}")

        print()
        print(f"PROMPT: {prompt}")
        
        os.system(f"python MusicLDM_cmdline.py \"{prompt}\" {audio_length_in_s} {ldm_path}")

        # COMMENT OUT THE FOLLOWING LINES TO SKIP MUSICGEN
        if music_gen is not False:
            music_gen = start_musicgen()
            generate_music(music_gen, prompt, music_path, audio_length_in_s)

    else:
        print()
        print(f"Music snippet exists for this prompt: {text_mus_id}")
    return text_mus_id


def get_n_most_frequent(text, n, POS):
    tokens = nltk.word_tokenize(text)
    # only keep adjectives
    tagged = nltk.pos_tag(tokens)
    words = [word for word, pos in tagged if pos == POS]
    # count the frequency of each adjective
    freq = Counter(words)
    # remove words that are smaller than 3 characters
    freq = {word: count for word, count in freq.items() if len(word) > 2 and word != "have"}
    # order freq by count
    freq = dict(sorted(freq.items(), key=lambda item: item[1], reverse=True))
    # get the n most frequent words
    most_common = ' '.join(list(freq.keys())[:n])
    return most_common


def epub_process(output_file_name, output_dir, generation_params, music_gen):
    """Process the epub book: download, extract text, and generate music"""

    paragraphs = generation_params["paragraphs"]
  
    book_id = os.path.basename(output_file_name).split('.')[0]

    # Create directory for music files if it doesn't exist
    music_dir = os.path.join(output_dir, 'music_from_MusicGenForTori')
    os.makedirs(music_dir, exist_ok=True)
    
    ind = 0
    for paragraph in paragraphs:
        chapter_id = f"chapter_{ind+1}"
        create_bg_music(paragraph, book_id, chapter_id, music_dir, generation_params, music_gen)
        ind += 1
    

def main(adj_flag, verb_flag, noun_flag, adjnoun_flag, most_frequent_n_words, first_n_chars, first_n_chapters, audio_length_in_s, prompt_intro, ollama_prompt, intro_in_ollama, HARDCODED_PROMPT, book, first_n_paragraphs, music_gen):
    # configure the output terminal to suppress warnings
    sys.stderr = None
    book_name = list(book.keys())[0]
    paragraphs = book[book_name]

    generation_params = {
        "adj_flag": adj_flag,
        "verb_flag": verb_flag,
        "noun_flag": noun_flag,
        "adjnoun_flag": adjnoun_flag,
        "most_frequent_n_words": most_frequent_n_words,
        "first_n_chars": first_n_chars,
        "first_n_chapters": first_n_chapters,
        "audio_length_in_s": audio_length_in_s,
        "prompt_intro": prompt_intro,
        "ollama_prompt": ollama_prompt,
        "intro_in_ollama": intro_in_ollama,
        "hardcoded_prompt": HARDCODED_PROMPT,
        "first_n_paragraphs": first_n_paragraphs,
        "paragraphs": paragraphs
    }

    print("hello")
    # Set the directory for saving downloaded books and music files locally
    books_dir = "./Generated_Music"  
    os.makedirs(books_dir, exist_ok=True)
    print("hello")

    with open('books_lookup.json', 'r') as file:
        data = json.load(file)
    
    print("hello")
    output_file_name = os.path.join(books_dir, f'{book_name}.epub')

    epub_process(output_file_name, books_dir, generation_params, music_gen)


if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # ADJUST PARAMETERS HERE
    # The same prompt generates different music snippets.
    # ----------------------------------------------------------------------

    music_gen = True

    true_flag_index = 4 # 0 for ADJ, 1 for VERB, 2 for NOUN, 3 for ADJ+NOUN pairs, 4 for first_n_paragraphs, NONE for no flag --> first_n_chars will be used
    most_frequent_n_words = 5
    adj_flag, verb_flag, noun_flag, adjnoun_flag, first_n_paragraphs_flag = (True if i == true_flag_index else False for i in range(5))

    first_n_paragraphs = 1
    if first_n_paragraphs_flag is False:
        first_n_paragraphs = None

    HARDCODED_PROMPT = ""
    
    # books should be a list of all the keys in the books_lookup.json file
    with open('all_book_chapters_mod.json', 'r') as file:
        data = json.load(file)
    booksers = ["the_three_taps", "grimms_fairy_tales", "huckleberry_finn", "jungle_book", "jane_eyre", "Five_weeks_in_the_baloon"]
    books = {key: value[:5] for key, value in data.items() if key.replace(".epub", "") in booksers}


    first_n_chars = 50
    first_n_chapters = 5

    audio_length_in_s = 30

    prompt_intro_flag = False
    prompt_intro = "instrumental music"
    if prompt_intro_flag is False:
        prompt_intro = ""

    ollama_flag = True
    ollama_prompt = "write a short, simple sentence starting with 'A ...' that describes music fitting this excerpt. Only output the musical description, without referring to this excerpt in any way: "
    if ollama_flag is False:
        ollama_prompt = ""

    intro_in_ollama = False

    for k, v in books.items():
        book = {k: v}
        main(adj_flag, verb_flag, noun_flag, adjnoun_flag, most_frequent_n_words, first_n_chars, first_n_chapters, audio_length_in_s, prompt_intro, ollama_prompt, intro_in_ollama, HARDCODED_PROMPT, book, first_n_paragraphs, music_gen)
        print()
        print("Done!")