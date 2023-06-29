import random
import pandas as pd
import numpy as np
import names
from random_address import real_random_address
import string



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
    bp_nr = "GC-4343.8188.1000"
    name = f"GC Customer-{random.randint(1000, 9999)}"
    personen_nr = f"{random.randint(1000, 9999)}.{random.randint(1000, 9999)}"
    konto_nr = f"{random.randint(1000, 9999)}.{random.randint(1000, 9999)}.{random.randint(1000, 9999)}"
    karten_nr = ""
    is_client = 1
    return [name, bp_nr, personen_nr, konto_nr, karten_nr, is_client]

def generateRandomStrings(num_strings):
    strings = []
    for _ in range(num_strings):
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

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

    string_count = 10000
    for _ in range(string_count):
        dataset.append(generateRandomStrings(5))

    # Shuffle the order of records
    random.shuffle(dataset)

    return dataset

# Example usage:
dataset = generate_random_dataset()

# Create DataFrame and save the dataset
df = pd.DataFrame(dataset, columns=["Name", "Address", "BP Nr", "Personen-Nr", "Konto-Nr", "Kartennummer", "isClient"])
df['isClient'] = df['isClient'].replace(np.nan,0)

#%%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the client data DataFrame

# Extract the input features (X) and target labels (y)
X = df[['Name', 'Address', "BP Nr", "Personen-Nr", "Konto-Nr", "Kartennummer"]].astype(str)  # Select the relevant columns as input features
y = df['isClient']  # Assuming 'IsClient' column indicates if it's client data or not

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer to convert text features into numerical vectors
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train.values.flatten())
X_test_vectors = vectorizer.transform(X_test.values.flatten())

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_vectors, y_train)

# Predict the labels for the test set
y_pred = model.predict(X_test_vectors)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Classify new input strings
new_data = pd.DataFrame({'Name': ['John Doe', 'Jane Smith'],
                         'Address': ['123 Main St', '456 Elm St'],
                         'OtherData': ['Some other data', 'More data']})
new_data_vectors = vectorizer.transform(new_data.values.flatten())
new_data_pred = model.predict(new_data_vectors)
print(f"New Data Predictions: {new_data_pred}")


#%%
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Sample string data
strings = [
    "Hello",
    "World",
    "Python",
    "Machine Learning"
    "6196.6187.1004"
]

# Convert strings to character-level features
vectorizer = CountVectorizer(analyzer="char")
X = vectorizer.fit_transform(df)

# Get the feature names (characters)
feature_names = vectorizer.get_feature_names()

# Convert the sparse matrix to a numpy array for easier manipulation
X_array = X.toarray()

# Print the character-level features
for i, string in enumerate(df):
    print(f"String: {df}")
    print("Character-Level Features:")
    for j, feature in enumerate(feature_names):
        if X_array[i, j] > 0:
            print(f"  {feature}: {X_array[i, j]} occurrence(s)")
    print("--------------------")

#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df.fillna("No Card", inplace=True)

df_combined = df[['Name', 'Address', "BP Nr", "Personen-Nr", "Konto-Nr", "Kartennummer"]]\
    .astype(str).apply(lambda x: " ".join(x), axis=1)

# Separate the features (text data) and the target variable
X = df_combined.values
y = df["isClient"].values

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