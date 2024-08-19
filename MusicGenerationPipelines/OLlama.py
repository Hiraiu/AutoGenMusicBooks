import requests
import json


def interact(prompt):
    url = "http://localhost:11434/api/generate"

    headers = {
        "Content-Type": "application/json",
    }

    data = {
        "model": "llama3",
        "prompt": f"{prompt}",
        "stream": False,
    }

    #models:
    # llama3, llama3:70b

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        pass
    else:
        print("Error: ", response.status_code, response.text)

    # return the part between "response" and "done"
    return response.text.split("response")[1].split("done")[0][3:-3]




if __name__ == "__main__":
    text = """
    Alice was beginning to get very tired of sitting by her sister on the
    bank, and of having nothing to do. Once or twice she had peeped into the
    book her sister was reading, but it had no pictures or conversations in
    it, "and what is the use of a book," thought Alice, "without pictures or
    conversations?”
    """
    ollama_prompt = "write a short, simple sentence starting with 'A ...' that describes music fitting this excerpt. Only output the musical description, without referring to this excerpt in any way: "
    
    print(interact(ollama_prompt + text))

