# Music Generation from Text Descriptions
This repository is part of my Master's thesis at the University of Zürich.

## Overview
This Python project generates music based on text descriptions using a combination of natural language processing and audio synthesis technologies. 
It utilizes two models: MusicGen and MusicLDM from the `transformers` to create music snippets that reflect the mood and content of the input text. 

## Requirements
- Python 3.8+
- Libraries: `transformers`, `nltk`, `ebooklib`, `BeautifulSoup`, `scipy`, `torch`

## Setup
All the necessary code for muisc generation is in the MusicGenerationPipelines folder. To generate music follow the next steps:
1. Clone this repository.
2. Install required Python packages:
   pip install -r requirements.txt
3. Install Ollama from this website: https://ollama.com/download 
4. Open a new system terminal and run "ollama run llama3" to establish a connection to Ollama
5. Run Pipeline_MusicLDM_and_MusicGen.py
6. You will now get new generated music from all the books in a folder named "Generated_Music" in the working directory. Along with the music file, there is also going to be a text file with the prompt used for generation and another text file that contains the paragraph that the promp was based on.
7. If you want to evaluate the code with CLAP scores, run the EVALUATE.py script from the EVALUATION folder.


## Adding new books for music generation 
Since the EPUB files format is not always consistent, to add new books the first paragraphs must be added manually in the all_book_chapters_mod.json. Alternatively, if you want to extract the first paragraph text from the EPUB directly, you can run "Alernate_Pipeline_MusicLDM_Musicgen", but some inconsitencies in the first paragraph extraction might appear depending on the book. Moreover, the book link must be added in books_lookup.json.
In the cases in which it cannot extract the first paragraph, it will extract the first 1000 characters and base the prompt on those.