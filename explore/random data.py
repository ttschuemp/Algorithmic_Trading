import random
import pandas as pd
import numpy as np
import names
import csv
from random_address import real_random_address
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# presidio Demo
# https://huggingface.co/spaces/presidio/presidio_demo


# https://github.com/f/awesome-chatgpt-prompts/blob/main/prompts.csv

# prompt generator
# https://flowgpt.com/prompt/jtD5PEhnhmEM_HADcTQny

# popular dataset for benchmarking QA engines,
# https://github.com/brmson/dataset-factoid-webquestions
# https://github.com/brmson/dataset-factoid-webquestions/blob/master/main/val.json

# google NQ datasets
# https://ai.google.com/research/NaturalQuestions

# https://www.kaggle.com/datasets/rtatman/questionanswer-dataset/code
# https://www.kaggle.com/code/leomauro/nlp-document-retrieval-for-question-answering

### bank client dataset
# https://data.world/lpetrocelli/retail-banking-demo-data/workspace/file?filename=CRM+Events.csv
# CRM data

# chatgtp prompts
# Read CSV file
with open("chatGTP_prompts.csv", encoding='utf-8') as fp:
    reader = csv.reader(fp, delimiter=',')
    # next(reader, None)  # skip the headers
    data_read = [row for row in reader]
    data_df = pd.DataFrame(data_read)
    chatgtp_prompts = data_df.astype(str).apply(lambda x: " ".join(x), axis=1)


# https://www.kaggle.com/datasets/rtatman/questionanswer-dataset/code
data_kaggle = pd.read_csv("random_question_data.txt", sep='\t', header=None)
data_kaggle_questions = data_kaggle.iloc[:,1]
data_kaggle_questions_df = pd.DataFrame(data_kaggle_questions)

def generate_private_customer():
    """
    logic:
    Level 1: Person             -> Vertragspartner: Max Mustermann, Personen-Nr: 1234.5678, Geburtsdatum
    Level 2: Business Partner   -> Definiert Geschäftsbeziehung, BP-Nr 0123.4567
    Level 3: Container          -> Standardportfolio, Depo, VV-Mandat, Vorsorge-Konto, 1234.5678.1001
    Level 4: Money Account      -> Konto, Kreditlimiten, Sicherheiten, Schrankfach, Mieterspardepot, Wertschriften
    Level 5: Asset              -> FX, Stocks, Bonds, Fonds, Options
    """
    name = names.get_full_name()
    bp_nr = f"{random.randint(1000, 9999)}.{random.randint(1000, 9999)}"
    addresse = f"{real_random_address()['address1']}, {real_random_address()['postalCode']}"
    konto_nr = f"{bp_nr}.100{random.randint(1, 9)}"
    personen_nr = f"{random.randint(1000, 9999)}.{random.randint(1000, 9999)}"
    karten_nr = str(random.randint(10**15, 10**16 - 1))  # 16-digit random number
    # container_nr = f"{random.randint(1000, 9999)}.{random.randint(1000, 9999)}.{random.randint(1000, 9999)}"
    is_client = 1
    return [name, addresse, bp_nr, personen_nr, konto_nr, karten_nr, is_client]

def generate_global_custody_customer():
    name = f"GC Customer-{random.randint(1000, 9999)}"
    bp_nr = "GC-4343.8188.1000"
    addresse = 'keine Addresse'
    personen_nr = f"{random.randint(1000, 9999)}.{random.randint(1000, 9999)}"
    konto_nr = f"{random.randint(1000, 9999)}.{random.randint(1000, 9999)}.{random.randint(1000, 9999)}"
    karten_nr = "keine"
    is_client = 1
    return [name, addresse, bp_nr, personen_nr, konto_nr, karten_nr, is_client]

def generateRandomStrings(num_strings):
    strings = []
    for _ in range(num_strings):
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1,30)))

        strings.append(random_string)

    return strings


def generate_random_dataset():
    dataset = []

    # Generate private customers (approximately 90% of the dataset)
    private_customers_count = 900
    for _ in range(private_customers_count):
        dataset.append(generate_private_customer())

    # Generate Global Custody customers (approximately 10% of the dataset)
    global_custody_customers_count = 100
    for _ in range(global_custody_customers_count):
        dataset.append(generate_global_custody_customer())

    # Shuffle the order of records
    random.shuffle(dataset)

    return dataset

def random_split(string):
    split_index = random.randint(1, len(string)-1)
    return string[:split_index], string[split_index:]

# Example usage:
dataset = generate_random_dataset()

# Create DataFrame and save the dataset
df = pd.DataFrame(dataset, columns=["Name", "Address", "BP Nr", "Personen-Nr", "Konto-Nr", "Kartennummer", "isClient"])
df.fillna("No Card", inplace=True)

chatgtp_prompts_df = pd.DataFrame(chatgtp_prompts)
chatgtp_prompts_df['isClient'] = 0
data_kaggle_questions_df['isClient'] = 0

# client classification
df_client = pd.concat([df['isClient'], chatgtp_prompts_df['isClient'], data_kaggle_questions_df['isClient']], ignore_index=True)

# combine question data
df_combined = df[['Name', 'Address', "BP Nr", "Personen-Nr", "Konto-Nr", "Kartennummer"]] \
    .astype(str).apply(lambda x: " ".join(x), axis=1)

df_combined = pd.concat([df_combined, chatgtp_prompts], ignore_index=True)
df_combined = pd.concat([df_combined, data_kaggle_questions_df.iloc[:,0]], ignore_index=True)
df_combined = pd.concat([df_combined, df_client], axis=1)

# random sample df
df_combined = df_combined.sample(frac = 1).reset_index(drop=True)

# remove nan
df_combined.dropna(axis=0, inplace=True)

#%%

# Separate the features (text data) and the target variable
X = df_combined.iloc[:,0].values
y = df_combined.iloc[:,1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Initialize and train a logistic regression model
model = LogisticRegression()
model.fit(X_train_vectors, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vectors)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


#%%
# Get user input
user_input = input("Enter the text to classify: ")

# Preprocess user input
user_input_combined = "".join(user_input)
user_input_vector = vectorizer.transform([user_input_combined])

# Classify user input
prediction = model.predict(user_input_vector)
if prediction[0] == 1:
    print("The text contains client information.")
else:
    print("The text does not contain client information.")

#%%

# https://stackabuse.com/text-classification-with-python-and-scikit-learn/

from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

stemmer = WordNetLemmatizer()
document = df_combined.iloc[:,0].apply(lambda x: ' '.join(([stemmer.lemmatize(word) for word in x.split()])))

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(document).toarray()

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# Get user input
user_input = input("Enter the text to classify: ")

# Preprocess user input
user_input_combined = "".join(user_input)
user_input_vector = vectorizer.transform([user_input_combined])

# Classify user input
prediction = classifier.predict(user_input_vector)
if prediction[0] == 1:
    print("The text contains client information.")
else:
    print("The text does not contain client information.")


#%%
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import openai

nltk.download('stopwords')
nltk.download('wordnet')

class TextClassifier:
    def __init__(self):
        self.stemmer = WordNetLemmatizer()
        self.vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
        self.tfidfconverter = TfidfTransformer()
        self.classifier = RandomForestClassifier(n_estimators=1000, random_state=0)

    def preprocess(self, documents):
        preprocessed_documents = documents.apply(lambda x: ' '.join([self.stemmer.lemmatize(word) for word in x.split()]))
        return preprocessed_documents

    def train(self, X, y):
        X_preprocessed = self.preprocess(X)
        X_vectorized = self.vectorizer.fit_transform(X_preprocessed).toarray()
        X_tfidf = self.tfidfconverter.fit_transform(X_vectorized).toarray()

        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=0)
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    def classify(self, text):
        text_combined = "".join(text)
        text_vector = self.vectorizer.transform([text_combined])
        prediction = self.classifier.predict(text_vector)

        if prediction[0] == 1:
            return "The text contains client information."
        else:
            return "The text does not contain client information."

    def generate_prompt(self, user_input):
        prompt = f"User: {user_input}\nAI:"
        self.response = openai.Completion.create(
            engine='davinci',
            prompt=prompt,
            max_tokens=50,
            temperature=0.7,
            n=1,
            stop=None,
            verify = False
        )
        return response.choices[0].text.strip()


#%%
def main():

    openai.api_key = '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'

    # Create an instance of the TextClassifier class
    classifier = TextClassifier()

    # Provide your training data (X) and corresponding labels (y)
    # Replace X and y with your own data
    X = df_combined.iloc[:, 0]
    y = df_combined.iloc[:, 1]

    # Train the classifier
    classifier.train(X, y)

    # Get user input
    user_input = input("Enter the text to classify: ")

    # Classify user input
    classification_result = classifier.classify([user_input])
    if classification_result != "The text contains client information.":
        print(classification_result)
        generated_prompt = classifier.generate_prompt(user_input)
        print("Generated Prompt:")
        print(generated_prompt)

    else:
        print(classification_result)

if __name__ == '__main__':
    main()




#%%
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import spacy.cli
spacy.cli.download("en_core_web_lg")

#%%

# https://pypi.org/project/presidio-anonymizer/


text="My client Hasan Inal has problems with his Account with Number 324.234.123"
# Set up the engine, loads the NLP module (spaCy model by default)
# and other PII recognizers
analyzer = AnalyzerEngine()
# Call analyzer to get results
results = analyzer.analyze(text=text,
                           entities=["PERSON","PHONE_NUMBER"],
                           language='en')

anonymizer = AnonymizerEngine()

anonymized_text = anonymizer.anonymize(text=text,analyzer_results=results)
#%%
