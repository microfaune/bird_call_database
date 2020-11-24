import numpy as np

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
        if self.global_score < 0.5:  # TO BE CHECKED
            return np.array([0]*self.expected_dim)

        redimensioned_local_score = self.reshape_prediction()
        return redimensioned_local_score

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

    def _reshape_small_array(self):
        """
        Simply fill the local_score with left zeros and right zeros

        :return: np.array of shape (self.expected_dim,)

        """
        right_padding = int((self.expected_dim - self.local_score.shape[0]) / 2)
        left_padding = self.expected_dim - self.local_score.shape[0] - right_padding
        return np.pad(self.local_score, (left_padding, right_padding), "constant", constant_values=(0, 0))

    def _reshape_big_array(self):
        range_to_check = self.local_score.shape[0] - self.expected_dim
        means = list()
        for i in range(range_to_check):
            means.append(np.mean(self.local_score[i:i+self.expected_dim]))
        max_index = means.index(max(means))
        return self.local_score[max_index:max_index+self.expected_dim]




