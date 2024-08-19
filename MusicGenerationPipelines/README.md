# Music Generation from Text Descriptions

## Overview
This Python project generates music based on text descriptions using a combination of natural language processing and audio synthesis technologies. 
It utilizes two models: MusicGen and MusicLDM from the `transformers` to create music snippets that reflect the mood and content of the input text.

## Features
- Text-to-music generation using pre-trained models.
- Flexible input options to specify text sources and musical styles.
- Output management to organize generated music files effectively.

## Requirements
- Python 3.8+
- Libraries: `transformers`, `nltk`, `ebooklib`, `BeautifulSoup`, `scipy`, `torch`

## Setup
1. Clone this repository.
2. Install required Python packages:
   pip install -r requirements.txt
3. Install Ollama from this website: https://ollama.com/download 
4. Open a new system terminal and run "ollama run llama3" to establish a connection to Ollama
5. Run Pipeline_MusicLDM_and_MusicGen.py
6. You will now get new generated music from all the books in a folder named "Generated_Music" in the working directory. Along with the music file, there is also going to be a text file with the prompt used for generation and another text file that contains the paragraph that the promp was based on.


## Adding new books to generate music for
Since the EPUB files format is not always consistent, to add new books the first paragraphs must be added manually in the all_book_chapters_mod.json. Alternatively, if you want to extract the text from the EPUB directly, you can run "Alernate_Pipeline_MusicLDM_Musicgen", but some inconsitencies in the first paragraph extraction might appear depending on the book.