import datetime
import os
import sys
import numpy as np
import json
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers.core import Flatten
from keras.layers import convolutional as conv
from keras.layers import pooling as pool
from keras.layers import normalization as norm
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Nadam
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_session(gpu_fraction=0.20):
    '''
    Set 15% of GPU space for the program
    '''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

REPLAY_FOLDER = sys.argv[1]
training_input = []
training_target = []

VISIBLE_DISTANCE = 7
width_img = 2 * VISIBLE_DISTANCE + 1
input_dim = 32 * VISIBLE_DISTANCE * VISIBLE_DISTANCE
np.random.seed(0)  # for reproducibility

feature_extractor = Sequential()
feature_extractor.add(conv.Convolution2D(32, 3, 3, border_mode='same', input_shape=(width_img, width_img, 4), init='glorot_normal'))
feature_extractor.add(PReLU(init='zero', weights=None))
feature_extractor.add(pool.MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
feature_extractor.add(conv.Convolution2D(64, 3, 3, border_mode='same', init='glorot_normal'))
feature_extractor.add(PReLU(init='zero', weights=None))
feature_extractor.add(pool.MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
feature_extractor.add(Flatten())

classifier = Sequential()
classifier.add(Dense(256, input_dim=feature_extractor.output_shape[1], init='glorot_normal'))
classifier.add(PReLU(init='zero', weights=None))
classifier.add(Dense(128, init='glorot_normal'))
classifier.add(PReLU(init='zero', weights=None))
classifier.add(Dense(5, init='glorot_normal'))

model = Sequential()
model.add(feature_extractor)
model.add(classifier)
optim = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(optim, 'categorical_crossentropy', metrics=['accuracy'])


def stack_to_input(stack, position):
    return np.transpose(
                np.take(np.take(stack,
                        np.arange(-VISIBLE_DISTANCE, VISIBLE_DISTANCE + 1)+position[0], axis=1, mode='wrap'),
                        np.arange(-VISIBLE_DISTANCE, VISIBLE_DISTANCE + 1)+position[1], axis=2, mode='wrap'), (2, 1, 0))


size = len(os.listdir(REPLAY_FOLDER))
for index, replay_name in enumerate(os.listdir(REPLAY_FOLDER)):
    if replay_name[-4:] != '.hlt':
        continue
    print('Loading {} ({}/{})'.format(replay_name, index, size))
    replay = json.load(open('{}/{}'.format(REPLAY_FOLDER, replay_name)))

    frames = np.array(replay['frames'])
    player = frames[:, :, :, 0]
    players, counts = np.unique(player[-1], return_counts=True)
    target_id = players[counts.argmax()]
    if target_id == 0:
        continue

    prod = np.repeat(np.array(replay['productions'])[np.newaxis], replay['num_frames'], axis=0)
    strength = frames[:, :, :, 1]

    moves = (np.arange(5) == np.array(replay['moves'])[:, :, :, None]).astype(int)[:128]
    stacks = np.array([player == target_id, (player != target_id) & (player != 0), prod/20, strength/255])
    stacks = stacks.transpose(1, 0, 2, 3)[:len(moves)].astype(np.float32)

    position_indices = stacks[:, 0].nonzero()
    sampling_rate = 1/stacks[:, 0].mean(axis=(1, 2))[position_indices[0]]
    sampling_rate *= moves[position_indices].dot(np.array([1, 10, 10, 10, 10]))  # weight moves 10 times higher than still
    sampling_rate /= sampling_rate.sum()
    sample_indices = np.transpose(position_indices)[np.random.choice(np.arange(len(sampling_rate)),
                                                                    min(len(sampling_rate), 2048), p=sampling_rate, replace=False)]

    replay_input = np.array([stack_to_input(stacks[i], [j, k]) for i, j, k in sample_indices])
    print(replay_input.shape)
    replay_target = moves[tuple(sample_indices.T)]

    training_input.append(replay_input.astype(np.float32))
    training_target.append(replay_target.astype(np.float32))

now = datetime.datetime.now()
tensorboard = TensorBoard(log_dir='./logs/'+now.strftime('%Y.%m.%d %H.%M'))
training_input = np.concatenate(training_input, axis=0)
training_target = np.concatenate(training_target, axis=0)
indices = np.arange(len(training_input))
np.random.shuffle(indices)  # shuffle training samples
training_input = training_input[indices]
training_target = training_target[indices]


model.fit(training_input, training_target, validation_split=0.2,
          callbacks=[EarlyStopping(patience=50),
                     ModelCheckpoint('model.h5', verbose=1, save_best_only=True),
                     tensorboard],
          batch_size=4096, nb_epoch=1000)

model = load_model('model.h5')

still_mask = training_target[:, 0].astype(bool)
print('STILL accuracy:', model.evaluate(training_input[still_mask], training_target[still_mask], verbose=0)[1])
print('MOVE accuracy:', model.evaluate(training_input[~still_mask], training_target[~still_mask], verbose=0)[1])
gc.collect()
