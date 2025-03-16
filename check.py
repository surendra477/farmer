from deep_translator import GoogleTranslator
translated_text = GoogleTranslator(source='auto', target='en').translate("Bonjour")
print(translated_text)  # "Hello"
