import numpy as np

from ..base import Model


class LookbackModel(Model):
    def __init__(self, look_back, look_forward=1):
        self.look_back = look_back
        self.look_forward = look_forward

    def _create_lookback_dataframe(self, df):
        """
        Take a time-series input, and turn it into two chunks - the "look back" and "look forward".

        Look back is what is fed to the network as input, and look forward is what the network is expected to predict.

        This is repeated, shifting by one for each timestep.
        """
        x, y = [], []
        for i in range(len(df) - self.look_back - self.look_forward):
            midpoint_idx = i + self.look_back
            shown_values = df[i:midpoint_idx]
            hidden_values = df[midpoint_idx : midpoint_idx + self.look_forward]
            x.append(shown_values)
            y.append(hidden_values)
        x = np.array(x)
        y = np.array(y)

        if self.look_forward == 1:
            y = np.reshape(y, (y.shape[0], y.shape[2]))

        return x, y

    def _pad_answer(self, df, result, value=0.0):
        num_points = len(df)
        final_answer = np.full(df.shape, value)
        final_answer[self.look_back : (num_points - self.look_forward)] = np.copy(
            result
        )
        return final_answer
