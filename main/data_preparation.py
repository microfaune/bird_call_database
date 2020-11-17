from microfaune.detection import RNNDetector
detector = RNNDetector()


class DataPreparation:

    def __init__(self):
        detector = RNNDetector()

    def predict_wav(self, wav_audio_path):
        global_score, local_score = detector.predict_on_wav(wav_audio_path)
        return global_score, local_score
