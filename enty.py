import hashlib

def encrypt_and_merge_keys_with_sha256(*keys):
    """
    Encrypt input keys combined into a single string using SHA256 and a predefined secret key.

    :param keys: Keys to be merged and encrypted.
    :return: SHA256 hash of the combined string with the secret.
    """
    secret = "RliqQVIVDcBW00CBEd5O5varL28XS1"
    # Combine all keys into a single string
    combined_keys = "".join(keys)
    # Combine the merged keys with the secret
    combined = combined_keys + secret
    # Create a SHA256 hash
    sha256_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()

    return sha256_hash

if __name__ == '__main__':
    # Example usage
    keys = ["7CB0AD9D-E682-46BF-A5A3-BBF0D357DBD6", "ios", "1.4.7", "1736996570978", "nat"]
    result = encrypt_and_merge_keys_with_sha256(*keys)
    print(f"-> 77f0de4b9900788c559b0f332cb424903b9313edddbdff24f208702291983a2d Encrypted: {result}")