import numpy as np
import librosa
import pickle
import keras
from keras.models import load_model
import streamlit as st
# Load the trained model, scaler, and encoder
def load_emotion_recognition_model(model_path, scaler_path, encoder_path):
    model = load_model(model_path)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    
    return model, scaler, encoder

def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    
    return result

def get_features_from_audio(file_path, chunk_size=2.5):
    data, sample_rate = librosa.load(file_path)
    total_duration = librosa.get_duration(y=data, sr=sample_rate)
    
    all_features = []

    for i in range(0, int(total_duration // chunk_size)):
        start = int(i * chunk_size * sample_rate)
        end = int((i + 1) * chunk_size * sample_rate)
        chunk = data[start:end]
        
        res1 = extract_features(chunk, sample_rate)
        all_features.append(res1)
    
    result = np.vstack(all_features)
    return result

def predict_emotion_from_audio(file_path, model, scaler, encoder):
    features = get_features_from_audio(file_path)
    scaled_features = scaler.transform(features)
    reshaped_features = np.expand_dims(scaled_features, axis=2)
    
    predictions = model.predict(reshaped_features)
    decoded_predictions = encoder.inverse_transform(predictions)
    
    unique, counts = np.unique(decoded_predictions, return_counts=True)
    final_prediction = unique[np.argmax(counts)]
    
    return final_prediction

def main():
    # Load the model, scaler, and encoder
    model, scaler, encoder = load_emotion_recognition_model('model/emotion_recognition_model.h5', 'model/scaler.pkl', 'model/encoder.pkl')

    # Streamlit UI
    st.title("Emotion Recognition from Audio")
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Predict emotion
        predicted_emotion = predict_emotion_from_audio("temp_audio.wav", model, scaler, encoder)
        
        st.write(f"Predicted Emotion: {predicted_emotion}")

if __name__ == "__main__":
    main()