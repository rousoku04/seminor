import numpy as np

import warnings
warnings.simplefilter('ignore', FutureWarning)

def prediction(model, input, output_length):
    len_input = len(input[0])
    # input : (delay + 1)
    for i in range(output_length):
        input = input.reshape(1, -1, 1)
        value_predict = model.predict(input[:, (i+1):, :])

        input = input.reshape(-1)
        input = np.append(input, value_predict)


    return input[len_input : ]