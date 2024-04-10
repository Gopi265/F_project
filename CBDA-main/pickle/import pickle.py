import pprint
import joblib

try:
    # Debug print statements
    print("Attempting to load TF-IDF Vectorizer...")
    tfidf_vectorizer = joblib.load('pickle/TFIDFvectorizer.pkl')
    print("TF-IDF Vectorizer loaded successfully.")

    print("Attempting to load Machine Learning Model...")
    machine_learning_model = joblib.load('pickle/bestmodel.pkl')
    print("Machine Learning Model loaded successfully.")

    # Print all attributes of the TF-IDF Vectorizer
    print("TF-IDF Vectorizer details:")
    pprint.pprint(tfidf_vectorizer.__dict__)

    # Print all attributes of the Machine Learning Model
    print("\nMachine Learning Model details:")
    pprint.pprint(machine_learning_model.__dict__)

except Exception as e:
    print("An error occurred:", e)
