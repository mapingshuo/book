import random
import paddle
import paddle.fluid as fluid
import numpy as np

CHANNEL_NUM = 3
W = 32
H = 32
#_mean = None
#_std = None
#_mean = np.asarray([[[0.49139969]], [[0.48215842]], [[0.44653093]]])
#_std = np.asarray([[[0.24703224]], [[0.24348513]], [[0.26158784]]])
_mean = np.asarray([[[0.485]], [[0.456]], [[0.406]]])
_std = np.asarray([[[0.229]], [[0.224]], [[0.225]]])

PAD_LEN=4

def cal_mean_and_std(reader):
    global _mean
    global _std
    element_sum = [0] * CHANNEL_NUM
    std_sum = [0] * CHANNEL_NUM

    # calculate _mean
    sample_count = 0
    for data, _ in reader():
        sample_count += 1
        data = data.reshape((CHANNEL_NUM, W, H))
        for i in range(CHANNEL_NUM):
            element_sum[i] += np.sum(data[i])
    _mean = [x / sample_count / W / H for x in element_sum]
    _mean = np.array(_mean).reshape((3, 1, 1))
   
    # calculate _std 
    for data, _ in reader():
        data = data.reshape((CHANNEL_NUM, W, H))
        for i in range(CHANNEL_NUM):
            std_sum[i] += np.sum(np.power(data[i] - _mean[i], 2))
    _std = [x / sample_count / W / H for x in std_sum]
    _std = np.sqrt(_std).reshape((3, 1, 1))

def normalize(reader):
    def iter():
	for data, label in reader():
            data = data.reshape((CHANNEL_NUM, W, H))
            data -= _mean
            data /= _std
            yield data, label
    return iter

def pad_zero(reader):
    def iter():
        for data, label in reader():
            data = np.concatenate((np.zeros((CHANNEL_NUM, H, PAD_LEN)),
				data,
				np.zeros((CHANNEL_NUM, H, PAD_LEN))), axis=2)
            data = np.concatenate((np.zeros((CHANNEL_NUM, PAD_LEN, W + 2 * PAD_LEN)),
				data,
				np.zeros((CHANNEL_NUM, PAD_LEN, W + 2 * PAD_LEN))), axis=1)
            yield data, label
    return iter

def crop(reader):
    """
    """
    def iter():
        for data, label in reader():
            w_start = random.randint(0, 2 * PAD_LEN - 1)
            h_start = random.randint(0, 2 * PAD_LEN - 1)
            data = data[:, w_start:w_start + W, h_start:h_start + H]
            yield data, label
    return iter

def mirror(reader):
    def iter():
        for data, label in reader():
            if random.randint(0, 1) == 0:
                yield data.reshape(CHANNEL_NUM * W * H), label
            else:
                #TODO: mirrored horizontally?
                yield data[:, :, ::-1].reshape(CHANNEL_NUM * W * H), label
    return iter

def preprocess(reader):
    print("Cifar-10 preprocess: norm, pad, crop, mirror")
    return mirror(crop(pad_zero(normalize(reader))))

if __name__=="__main__": 
    reader = paddle.dataset.cifar.train10()
    #cal_mean_and_std(reader)
    _mean = np.asarray([[[0.49139969]], [[0.48215842]], [[0.44653093]]])
    _std = np.asarray([[[0.24703224]], [[0.24348513]], [[0.26158784]]])
    print("mean: ", _mean)
    print("std: ", _std)
    norm_reader = normalize(reader)
    pad_reader = pad_zero(norm_reader)
    crop_reader = crop(pad_reader)
    mirrored_reader = mirror(crop_reader)
    c = 0
    for a in mirrored_reader():
        c += 1
