import random
import pandas as pd
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
    is_client = True
    return [name, addresse, bp_nr, personen_nr, konto_nr, karten_nr, is_client]

def generate_global_custody_customer():
    bp_nr = "GC-4343.8188.1000"
    name = f"GC Customer-{random.randint(1000, 9999)}"
    personen_nr = f"{random.randint(1000, 9999)}.{random.randint(1000, 9999)}"
    konto_nr = f"{random.randint(1000, 9999)}.{random.randint(1000, 9999)}.{random.randint(1000, 9999)}"
    karten_nr = ""
    is_client = True
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
df = pd.DataFrame(dataset)

#%%
