from fastapi import FastAPI, UploadFile, File, HTTPException
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2
import io
from typing import Dict, Tuple
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import MidTermFeatures as aF
from pydub import AudioSegment

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://mood-based-doorbell.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def analyze_emotion(image_data: bytes) -> Tuple[str, float]:
    VALID_EMOTIONS = {'happy', 'sad', 'angry'}

    # Convert bytes to image
    image = Image.open(io.BytesIO(image_data))
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        image.save(temp_file, format='JPEG')
        temp_path = temp_file.name

    try:
        result = DeepFace.analyze(
            temp_path,
            actions=['emotion'],
            enforce_detection=True,
            detector_backend='retinaface'
        )
        
        emotions = result[0]['emotion']
        
        # Filter only our three emotions
        relevant_emotions = {
            emotion: emotions[emotion]
            for emotion in VALID_EMOTIONS
        }
        
        print("Emotion scores:", relevant_emotions)
        
        # Always return the emotion with highest confidence
        max_emotion, confidence = max(relevant_emotions.items(), key=lambda x: x[1])
        print(f"Selected emotion: {max_emotion} ({confidence:.2f}%)")
        
        return max_emotion, confidence
    
    except Exception as e:
        print(f"Error in emotion analysis: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail="Failed to analyze emotion in image"
        )
    
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

async def train_voice_model():
    """
    Train a voice emotion classifier using sample audio files.
    Requires audio files organized in directories by emotion.
    """
    try:
        # Create directories for training data if they don't exist
        base_dir = "training_data"
        emotions = ['happy', 'sad', 'angry']
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            for emotion in emotions:
                os.makedirs(os.path.join(base_dir, emotion))
            print("Created training directories. Please add .wav files to:")
            for emotion in emotions:
                print(f"- {os.path.join(base_dir, emotion)}/")
            return False

        # Check if we have training data
        has_data = all(
            len(os.listdir(os.path.join(base_dir, emotion))) > 0
            for emotion in emotions
        )
        
        if not has_data:
            print("Please add training audio files first!")
            return False

        # Train the model
        dirs = [os.path.join(base_dir, emotion) for emotion in emotions]
        aT.extract_features_and_train(
            dirs,
            1.0, 1.0, 0.05, 0.05,
            "svm",
            "emotion_model",
            False
        )
        print("Model trained successfully!")
        return True

    except Exception as e:
        print(f"Error training model: {str(e)}")
        return False

async def analyze_voice_emotion(audio_data: bytes) -> str:
    # First ensure we have a trained model
    if not os.path.exists("emotion_model"):
        model_trained = await train_voice_model()
        if not model_trained:
            raise HTTPException(
                status_code=500,
                detail="Voice emotion model not trained. Please add training data first."
            )

    try:
        # Convert audio to WAV format
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                audio.export(temp_file.name, format='wav')
                temp_path = temp_file.name
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="Invalid audio format. Please provide a valid audio file."
            )

        # Extract features with correct parameters
        [classifier_output] = aT.file_classification(
            temp_path,
            "emotion_model",
            "svm"
        )
        
        # Map the output to emotions
        emotion_map = {0: 'happy', 1: 'sad', 2: 'angry'}
        predicted_emotion = emotion_map[int(classifier_output)]
        
        return predicted_emotion

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail="Failed to analyze voice emotion"
        )
    
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/detect-mood")
async def detect_mood(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Detect the dominant mood (happy, sad, or angry) from an uploaded image.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
        
    try:
        image_data = await file.read()
        mood, confidence = await analyze_emotion(image_data)
        return {
            "mood": mood,
            "confidence": f"{confidence:.2f}%"
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in detect_mood: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/detect-voice-mood")
async def detect_voice_mood(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Detect the dominant mood (happy, sad, or angry) from an uploaded voice recording.
    Accepts various audio formats (MP3, WAV, OGG, etc.)
    """
    if not file.content_type.startswith('audio/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an audio file"
        )
    
    try:
        audio_data = await file.read()
        mood = await analyze_voice_emotion(audio_data)
        return {"mood": mood}
    
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in detect_voice_mood: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze voice emotion"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)