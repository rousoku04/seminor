import tensorflow as tf
import numpy as np

import warnings
warnings.simplefilter('ignore', FutureWarning)

# もしも損失関数を追加したい場合、このコードに関数を追加したのちに、
# from loss_function import 損失関数の名前
# とすれば　main.pyでその損失関数を使うことができます。


def pinball_loss_alpha(alpha: float):
    # alpha \in (0, 1)
    def pinball_loss(y_true, y_pred):
        error = y_true - y_pred

        one = tf.ones(tf.shape(error))

        error_temp_1 = tf.where(error > 0, alpha * one, one)
        error_temp_2 = tf.where(error < 0, (alpha - 1) * one, one)

        loss = error * error_temp_1 * error_temp_2

        return tf.reduce_mean(loss)

    return pinball_loss