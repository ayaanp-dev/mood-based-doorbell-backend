import os
import shutil
from pathlib import Path

def organize_ravdess_data():
    # Emotion mapping in RAVDESS:
    # 03 = happy
    # 04 = sad
    # 05 = angry
    emotion_map = {
        '03': 'happy',
        '04': 'sad',
        '05': 'angry'
    }
    
    source_dir = "Audio_Speech_Actors_01-24"
    target_dir = "training_data"
    
    # Create target directories
    for emotion in emotion_map.values():
        os.makedirs(os.path.join(target_dir, emotion), exist_ok=True)
    
    # Process all audio files
    for actor_dir in Path(source_dir).glob("Actor_*"):
        for audio_file in actor_dir.glob("*.wav"):
            # RAVDESS filename format: 03-01-05-01-01-01-01.wav
            # where the third number (05) is the emotion code
            emotion_code = audio_file.stem.split("-")[2]
            
            if emotion_code in emotion_map:
                emotion = emotion_map[emotion_code]
                target_path = os.path.join(target_dir, emotion, audio_file.name)
                shutil.copy2(audio_file, target_path)
                print(f"Copied {audio_file.name} to {emotion}")

if __name__ == "__main__":
    organize_ravdess_data()