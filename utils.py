from config import voc_path
import pickle


# Save questions and answers vocabularies in pickle format
def save_voc(questionswords2int, answerswords2int, answersint2words):
    voc = {
        "questionswords2int": questionswords2int,
        "answerswords2int": answerswords2int,
        "answersint2words": answersint2words
    }
    try:
        pickle.dump(voc, open(voc_path, "wb"))
        print("Vocabularies saved")
    except Exception as e:
        print("Failed to save vocabularies : " + str(e))


# Load vocabularies
def load_voc():
    try:
        voc = pickle.load(open(voc_path, "rb"))
        return voc["questionswords2int"], voc["answerswords2int"], voc["answersint2words"]
    except Exception as e:
        print("Failed to load preprocessing parameters : " + str(e))
