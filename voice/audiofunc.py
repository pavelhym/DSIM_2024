

import librosa
import joblib
import numpy as np
from librosa.feature import melspectrogram, mfcc, rms, zero_crossing_rate, chroma_stft
import numpy as np

def root_mean_square(input, tsize = 150):
  root = rms(y = input *1.0)
  root = root[:, 0:min(tsize, root.shape[1])]
  root = np.pad(root, ((0,0),(0,tsize - root.shape[1])))
  output = np.ravel(root)
  return output

def zcr(input, tsize = 150):
  zero = zero_crossing_rate(y = input *1.0)
  zero = zero[:, 0:min(tsize, zero.shape[1])]
  zero = np.pad(zero, ((0,0),(0,tsize - zero.shape[1])))
  output = np.ravel(zero)
  return output

def chroma(input, rate, tsize = 150):
  chroma = chroma_stft(y = input *1.0, sr=rate)
  chroma = chroma[:, 0:min(tsize, chroma.shape[1])]
  chroma = np.pad(chroma, ((0,0),(0,tsize - chroma.shape[1])))
  output = np.ravel(chroma)
  return output

def feats_mel(input, rate, tsize=150):
    mel_spec = melspectrogram(y=input*1.0, sr=rate) # Extracting the features of the Mel spectrogram
    mel_spec = mel_spec[:, 0:min(tsize, mel_spec.shape[1])] # Considering only the first 'tsize = 10' temporal columns
    mel_spec = np.pad(mel_spec, ((0,0),(0,tsize - mel_spec.shape[1])), 'constant', constant_values=(0)) # Adding zeros in order to have the same number of columns for each audio track
    output = np.ravel(mel_spec)
    return output

def feats_mfcc(input, rate, tsize=150):
    mfccs = mfcc(y=input*1.0, sr=rate, n_mfcc = 40) # Extracting the MFCC coefficients of the audio track
    mfccs = mfccs[:, 0:min(tsize, mfccs.shape[1])] # Considering only the first 'tsize = 10' temporal columns
    mfccs = np.pad(mfccs, ((0,0),(0,tsize - mfccs.shape[1])), 'constant', constant_values=(0)) # Adding zeros in order to have the same number of columns for each audio track
    output = np.ravel(mfccs)
    return output

def combo(input, rate):
    return np.concatenate((root_mean_square(input),
                           zcr(input),
                           chroma(input, rate=rate),
                           feats_mel(input, rate = rate),
                           feats_mfcc(input, rate = rate)))

def extract_features(audio, sr):
  features = []
  features.append(combo(audio, rate = sr))
  eps = 0.001
  features = np.array(features)
  features_mean = features.mean(axis=1)
  features_std = features.std(axis=1)
  features = (features - features_mean + eps)/(features_std + eps)
  features = [row for row in features]

  return np.array(features)




def predict(audio, sr):
  clf = joblib.load('voice/cnn_model.joblib')
  features=extract_features(audio,sr)
  return clf.predict(features)





def detect(array):
  max = np.argmax(array)
  if max == 0: return 'Angry :rage:'
  if max == 1: return 'Calm :neutral_face:'
  if max == 2: return 'Disgust :woozy_face:'
  if max == 3: return 'Fearful :fearful:'
  if max == 4: return 'Happy :blush:'
  if max == 5: return 'Neutral :no_mouth:'
  if max == 6: return 'Sad :sleepy:'
  if max == 7: return 'Surprised :dizzy_face:'
  if len(array) != 8: return 'The lenght of the array is more than 8'