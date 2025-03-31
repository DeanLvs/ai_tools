import random
import pandas as pd

# Function to generate a random alphanumeric string
def generate_random_string():
    letters = random.sample('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 6)
    numbers = random.sample('0123456789', 4)
    return ''.join(f"{l}{n}" for l, n in zip(letters, numbers))

# Function to generate a random 10-digit number
def generate_random_number():
    return ''.join(random.choices('01234567', k=10))

# Generate 100 rows of data and print them in the desired format
data = [{'Code': generate_random_string(), 'Number': generate_random_number()} for _ in range(100)]

# Print each item in the required format
output = '\n'.join(f"{item['Code']} {item['Number']}" for item in data)
print(output)
