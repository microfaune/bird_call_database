from microfaune.detection import RNNDetector
detector = RNNDetector()


class DataPreparation:

    def __init__(self, expected_dim):
        self.detector = RNNDetector()
        self.expected_dim = expected_dim
        self.global_score = None
        self.local_score = None

    def predict_wav(self, wav_audio_path):
        self.global_score, self.local_score = self.detector.predict_on_wav(wav_audio_path)
        redimension_local_score = self.reshape_prediction(self.local_score)
        return redimension_local_score

    def reshape_prediction(self):
        """
        If dimension is too small for the model, we will add some zeros to fill the blanks
        If dimension is too high, we will keep the most interesting portion.
        :return:
        """
        if self.local_score.shape[0] < self.expected_dim:
            return self._reshape_small_array()
        elif self.local_score.shape[0] > self.expected_dim:
            return self._reshape_big_array()
        return self.local_score

    def _reshape_small_array(self, local_score):
        pass

    def _reshape_big_array(self, local_score):
        pass
