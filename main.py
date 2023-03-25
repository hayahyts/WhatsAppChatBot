import re
from sklearn.model_selection import train_test_split


def read_chat_data(file_path):
    with open(file_path, "r") as file:
        chat_data = file.read()

    messages = re.findall(r"\[.*?\] (.*?): (.*)", chat_data)

    # Define a list of unwanted texts
    unwanted_texts = ["No one outside of this chat, not even WhatsApp"]

    # Filter out tuples containing any of the unwanted texts from the messages list
    messages = [(user, message) for user, message in messages if not any(text in message for text in unwanted_texts)]

    input_output_pairs = []
    current_user, current_message = messages[0]

    for i in range(1, len(messages)):
        next_user, next_message = messages[i]

        if "image omitted" in next_message:
            next_message = "sent a photo."
        elif "video omitted" in next_message:
            next_message = "sent a video."
        elif "sticker omitted" in next_message:
            next_message = "sent a sticker."
        elif "document omitted" in next_message:
            next_message = "sent a document."
        elif "audio omitted" in next_message:
            next_message = "sent an audio"

        if next_user == current_user:
            current_message += ". " + next_message
        else:
            input_output_pairs.append((current_message, next_message))
            current_user, current_message = next_user, next_message

    return input_output_pairs


def write_pairs_to_file(pairs, output_path):
    with open(output_path, "w") as file:
        file.truncate(0)
        for pair in pairs:
            file.write(f"Input: {pair[0]}\nOutput: {pair[1]}\n")


def split_data(file_path):
    with open(file_path, "r") as file:
        input_output_pairs = [line.strip().split("\t") for line in file.readlines()]

    # Split the dataset into training, validation, and testing sets
    train_val_pairs, test_pairs = train_test_split(input_output_pairs, test_size=0.2, random_state=42)
    train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=0.2, random_state=42)

    # Save the training, validation, and testing sets to separate files
    with open("train_pairs.txt", "w") as file:
        file.truncate(0)
        for pair in train_pairs:
            file.write(pair[0] + "\t" + pair[1] + "\n")

    with open("val_pairs.txt", "w") as file:
        file.truncate(0)
        for pair in val_pairs:
            file.write(pair[0] + "\t" + pair[1] + "\n")

    with open("test_pairs.txt", "w") as file:
        file.truncate(0)
        for pair in test_pairs:
            file.write(pair[0] + "\t" + pair[1] + "\n")


def train_model():
    file_path = "beb_chat.txt"  # Replace with the path to your text file
    output_path = "input_output_pairs.txt"  # Replace with the path to the output file

    input_output_pairs = read_chat_data(file_path)
    write_pairs_to_file(input_output_pairs, output_path)
    split_data(output_path)


if __name__ == '__main__':
    train_model()
