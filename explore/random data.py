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


# https://github.com/f/awesome-chatgpt-prompts/blob/main/prompts.csv

# prompt generator
# https://flowgpt.com/prompt/jtD5PEhnhmEM_HADcTQny

# popular dataset for benchmarking QA engines,
# https://github.com/brmson/dataset-factoid-webquestions
# https://github.com/brmson/dataset-factoid-webquestions/blob/master/main/val.json

#this!
# https://ai.google.com/research/NaturalQuestions

# https://www.kaggle.com/datasets/rtatman/questionanswer-dataset/code

# chatgtp prompts
# Read CSV file
with open("chatGTP_prompts.csv", encoding='utf-8') as fp:
    reader = csv.reader(fp, delimiter=',')
    # next(reader, None)  # skip the headers
    data_read = [row for row in reader]
    data_df = pd.DataFrame(data_read)
    chatgtp_prompts = data_df.astype(str).apply(lambda x: " ".join(x), axis=1)


data_kaggle = pd.read_csv("random_question_data.txt", sep='\t')


def generate_private_customer():
    name = names.get_full_name()
    bp_nr = f"{random.randint(1000, 9999)}.{random.randint(1000, 9999)}"
    addresse = f"{real_random_address()['address1']}, {real_random_address()['postalCode']}"
    konto_nr = f"{bp_nr}.100{random.randint(1, 9)}"
    personen_nr = f"{random.randint(1000, 9999)}.{random.randint(1000, 9999)}"
    karten_nr = str(random.randint(10**15, 10**16 - 1))  # 16-digit random number
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

    #string_count = 10000
    #for _ in range(string_count):
    #    dataset.append(generateRandomStrings(5))

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

chatgtp_prompts_df = pd.DataFrame(chatgtp_prompts)
chatgtp_prompts_df['isClient'] = 0

#df['isClient'] = df['isClient'].replace(np.nan,0)

df.fillna("No Card", inplace=True)

#df_combined = df[['Name', 'Address', "BP Nr", "Personen-Nr", "Konto-Nr", "Kartennummer"]]\
#    .astype(str).apply(lambda x: " ".join(x), axis=1)


df_combined = df[['Name', 'Address', "BP Nr", "Personen-Nr", "Konto-Nr", "Kartennummer"]] \
    .astype(str).apply(lambda x: " ".join(x), axis=1)

df_combined = df_combined.append(chatgtp_prompts).to_frame()

df_combined["isClient"] = df["isClient"].append(chatgtp_prompts_df['isClient'])


df_combined = df_combined.sample(frac = 1).reset_index(drop=True)

# Apply random_split function to each row of the 'Text' column
#df_combined = df_combined.apply(lambda x: pd.Series(random_split(x))).astype(str).apply(lambda x: " ".join(x), axis=1)

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
#