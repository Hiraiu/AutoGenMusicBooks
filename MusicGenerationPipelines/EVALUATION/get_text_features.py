from transformers import AutoTokenizer, ClapModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def get_text_features(texts):
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)
    return text_features
