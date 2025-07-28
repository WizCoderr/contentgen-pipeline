from moviepy.editor import *

def create_video(images, voice_path, output="media/final_video.mp4"):
    clips = [ImageClip(img).set_duration(3) for img in images]
    video = concatenate_videoclips(clips, method="compose")
    audio = AudioFileClip(voice_path)
    final = video.set_audio(audio)
    final.write_videofile(output, fps=24)
    return output
