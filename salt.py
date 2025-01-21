import random
import string

def hash_function(seed):
    hash = 0
    for c in seed:
        hash = (hash * 131 + ord(c)) % (2**32)  # Simulate 32-bit unsigned int overflow
    return hash

def generate_salt(desired_value, mod_value=10, salt_length=256):
    characters = string.ascii_letters + string.digits + string.punctuation
    attempt = 0
    while True:
        attempt += 1
        initial_salt = ''.join(random.choices(characters, k=salt_length - 1))
        # Compute the hash up to this point
        partial_hash = 0
        for c in initial_salt:
            partial_hash = (partial_hash * 131 + ord(c)) % (2**32)
        # Find the last character that results in the desired modulo
        for c in characters:
            total_hash = (partial_hash * 131 + ord(c)) % (2**32)
            if total_hash % mod_value == desired_value:
                full_salt = initial_salt + c
                print(f"Salt found after {attempt} attempts.")
                return full_salt
        # If no suitable character is found, retry with a new initial_salt

# Desired MAX_CONCURRENT_JOBS value
desired_max_jobs = 1  # Set your desired value here

# Generate the salt
salt = generate_salt(desired_max_jobs, mod_value=10, salt_length=256)

if salt:
    print(f"Generated salt of length {len(salt)} for MAX_CONCURRENT_JOBS = {desired_max_jobs}")
    print(f"Salt: '{salt}'")
    # Verify the hash function
    final_hash = hash_function(salt)
    print(f"Hash: {final_hash}")
    print(f"Hash % 10: {final_hash % 10}")
else:
    print("No suitable salt found with the given parameters.")
