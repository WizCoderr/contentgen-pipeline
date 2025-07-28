from gtts import gTTS

def generate_voiceover(script):
    tts = gTTS(text=script, lang='en')
    output_path = f"media/{script[:10]}.mp3"
    tts.save(output_path)
    return output_path


if __name__ == "__main__":
    generate_voiceover(script="HelloWorld")