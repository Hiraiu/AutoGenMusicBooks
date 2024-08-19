from evaluate_further_experiments import evaluate_further_experiments
from evaluate_initial_experiments import evaluate_initial_experiments
from compare_text_and_audio_features import evaluate_similarity

def evaluate(caption_text_flag, char_limit, paragraph_limit, evaluate_further_flag, eval_function, path_to_eval, model):

        output_file = "eval_further_experiments"
        if model == "MusicGen":
            output_file += "_musicgen"
        elif model == "MusicLDM":
            output_file += "_musicldm"
        if caption_text_flag:
            output_file += "_caption"
        if char_limit:
            output_file += f"_{char_limit}"
        # add the variable name of the evaluation function
        output_file += f"_{eval_function.__name__}"
        output_file = output_file.replace("evaluate_similarity", "clap")

        if evaluate_further_flag:
            output_file = "EVAL_FURTHER/" + output_file
            evaluate_further_experiments(path_to_eval, output_file, caption_text_flag, eval_function, paragraph_limit, char_limit)

        else:
            output_file = "EVAL_INITIAL/" + output_file
            output_file = output_file.replace("further", "initial")
            evaluate_initial_experiments(output_file, caption_text_flag, model, eval_function, char_limit)



if __name__ == "__main__":
        # PARAMETERS
        # ------------------------------------------
        # ------------------------------------------

        char_limit = None
        paragraph_limit = 1

        evaluate_further_flag = True

        eval_function = evaluate_similarity

        # SPECIFY THE PATH TO THE FILES TO BE EVALUATED
        path_to_eval = "../Generated_Music_directly_from_Epubs/music_from_MusicGenForTori/first_n_paragraphs/1_paragraphs/ollama/other"
        model = "MusicGen"
        # ------------------------------------------
        # ------------------------------------------


        for caption_text_flag in [False, True]:
            evaluate(caption_text_flag, char_limit, paragraph_limit, evaluate_further_flag, eval_function, path_to_eval, model)