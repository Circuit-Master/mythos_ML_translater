import json
import readline

print("Translation Tool")

with open("message log.json", 'r') as file:
	message_data = json.load(file)

with open("data.json", 'r') as file:
	translated_data = json.load(file)

messages_to_translate = set(message_data) - set(map(lambda e: e["input"], translated_data))

print("Enter to submit, CTRL+C to exit, leave blank to skip")

for message in messages_to_translate:
	print(f"\nMessage: {message}")

	translation = input("Your translation: ")

	if translation:
		translated_data.append({"input": message, "output": translation})

		with open("data.json", 'w') as file:
			json.dump(translated_data, file, ensure_ascii=False, indent=4)
	else:
		print("Skipped")
