import argparse
import numpy as np
import tensorflow as tf
import os
import sys
from models import pointnet_cls
import data_provider



parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
# GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
LOG_DIR = FLAGS.log_dir
# 记录log信息
Log_out = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
Log_out.write(str(FLAGS) + '\n')

# ModelNet40 official train/test split
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILES = data_provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = data_provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

def log_string(out_str):
    Log_out.write(out_str+'\n')
    Log_out.flush()
    print(out_str)


def train_cls():
    model = pointnet_cls.ClsModel(NUM_POINT)
    model.build(input_shape=(None, 1024, 3))
    model.summary()

    optimizer = tf.keras.optimizers.Adam(0.001)
    for epoch in range(MAX_EPOCH):
        log_string('**** EPOCH %03d ****' % (epoch))
        train_file_idx = np.arange(0, len(TRAIN_FILES))
        np.random.shuffle(train_file_idx)
        for i in range(len(TRAIN_FILES)):
            pc_data, pc_label = data_provider.load_h5(TRAIN_FILES[train_file_idx[i]])
            pc_data = pc_data[:, 0:NUM_POINT, :]
            pc_data, pc_label, _ = data_provider.shuffle_data(pc_data, pc_label)

            pc_label = tf.squeeze(pc_label)
            pc_label = tf.cast(pc_label, dtype=tf.int64)
            pc_label = tf.one_hot(pc_label, depth=40)

            print(pc_label.shape)
            data_size = pc_data.shape[0]
            num_batches = data_size // BATCH_SIZE

            total_correct = 0
            loss_sum = 0
            total_seen = 0
            for batch in range(num_batches):
                start = batch * BATCH_SIZE
                end = (batch + 1) * BATCH_SIZE
                # augument data
                rotated_data = data_provider.rotate_point_cloud(pc_data[start:end, :, :])
                jitted_data = data_provider.jitter_point_cloud(rotated_data)
                jitted_label = pc_label[start:end]
                jitted_data = tf.cast(jitted_data, dtype=tf.float32)
                with tf.GradientTape() as tape:
                    out, end_points = model(jitted_data)
                    loss = get_loss(out, jitted_label, end_points)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                ## caluculate
                pred_val = tf.argmax(out, axis=1)
                jitted_label = tf.argmax(jitted_label, axis=1)
                correct = tf.reduce_sum(tf.cast(tf.equal(pred_val, jitted_label), dtype=tf.int64))
                print(correct)
                total_correct += correct
                total_seen += BATCH_SIZE

                # loss_sum += loss
                log_string('Batch%d mean_loss:%f accuracy:%f' %(batch, loss, float(total_correct) / float(total_seen)))


def preprocessing(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int64)
    y = tf.one_hot(y, depth=40)
    y = tf.squeeze(y, axis=0)
    return (x, y)

def train_cls2():

    model = pointnet_cls.ClsModel(NUM_POINT)
    model.build(input_shape=(None, 1024, 3))
    # model.summary()
    ## train_data
    pc_data, pc_label = data_provider.load_h5(TRAIN_FILES[0])
    pc_data = pc_data[:, 0:NUM_POINT, :]
    pc_data, pc_label, _ = data_provider.shuffle_data(pc_data, pc_label)
    all_data, all_label = pc_data, pc_label
    for i in range(1, len(TRAIN_FILES)):
        pc_data, pc_label = data_provider.load_h5(TRAIN_FILES[i])
        pc_data = pc_data[:, 0:NUM_POINT, :]
        pc_data, pc_label, _ = data_provider.shuffle_data(pc_data, pc_label)
        all_data = np.concatenate((all_data,pc_data), axis=0)
        all_label = np.concatenate((all_label, pc_label), axis=0)
    ##
    val_data, val_label = data_provider.load_h5(TEST_FILES[0])
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_label))
    val_dataset = val_dataset.map(preprocessing)

    dataset = tf.data.Dataset.from_tensor_slices((all_data, all_label))
    dataset = dataset.map(preprocessing).shuffle(10000).batch(32).repeat()
    op = tf.keras.optimizers.Adam(0.0001)
    # for (x, y) in iter(dataset):
    #     with tf.GradientTape() as tape:
    #         out = model(x)
    #         loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_true=y, y_pred=out, from_logits=True))
    #     grads = tape.gradient(loss, model.trainable_variables)
    #     # print(grads)
    #     op.apply_gradients(zip(grads, model.trainable_variables))
    #     pred = tf.argmax(out, axis=1)
    #     y_label = tf.argmax(y, axis=1)
    #
    #     correct = tf.reduce_sum(tf.cast(tf.equal(pred, y_label),dtype=tf.int64))
    #     print('loss: %4f  acc: %.2f ' %( loss, int(correct) / y.shape[0]))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.categorical_crossentropy,\
                  metrics=['accuracy'])
    model.fit(dataset, steps_per_epoch=150, epochs=50, validation_data=val_dataset, validation_steps=200)
    model.save_weights('./checkpoints/net.ckpt')


def get_loss(pred, label, end_points, reg_weights=0.001):
    loss = tf.losses.categorical_crossentropy(y_pred=pred, y_true=label)
    cls_loss = tf.reduce_mean(loss)

    tf.summary.scalar('cls_loss', cls_loss)
    ## transform
    transform = end_points['transform']
    K = transform.shape[1]
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0, 2, 1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    tf.summary.scalar('mat_loss', mat_diff_loss)
    return cls_loss + mat_diff_loss * reg_weights



if __name__ == '__main__':
    train_cls2()
