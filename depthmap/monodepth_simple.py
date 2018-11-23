from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import numpy as np
import argparse
import tensorflow as tf
import scipy.misc
import matplotlib.pyplot as plt
import cv2
from depthmap.monodepth_model import monodepth_parameters, MonodepthModel

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

checkpoint_path = '/home/shreyas/Code/target-tracker/depthmap/models/model_kitti'
input_height = 256
input_width = 512
encoder = 'vgg'

sess = None
model = None
left = None

params = monodepth_parameters(
    encoder='vgg',
    height=300,
    width=300,
    batch_size=2,
    num_threads=8,
    num_epochs=1,
    do_stereo=False,
    wrap_mode="border",
    use_deconv=False,
    alpha_image_loss=0,
    disp_gradient_loss_weight=0,
    lr_loss_weight=0,
    full_summary=False)


def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def replace_pixels(margin, env):
    max_dp = np.max(env)
    env[:, -margin:] = max_dp
    env[:, :margin] = max_dp
    env[:margin, :] = max_dp
    env[-margin:, :] = max_dp

    return env


def init():
    global sess, model , left
    left = tf.placeholder(tf.float32, [2, input_height, input_width, 3])
    model = MonodepthModel(params, "test", left, None)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()

    # RESTORE
    restore_path = checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)


def generate_depth_map_frames(params, frames, video_path):
    global sess, left, model
    count = 0
    # Process Video here
    output_directory = os.path.dirname(video_path)
    output_name = os.path.splitext(os.path.basename(video_path))[0]

    try:
        os.mkdir(os.path.join(output_directory, 'disparity'))
    except:
        pass

    for input_image in frames:
        original_height, original_width, num_channels = input_image.shape
        input_image = scipy.misc.imresize(input_image, [input_height, input_width], interp='lanczos')
        input_image = input_image.astype(np.float32) / 255
        input_images = np.stack((input_image, np.fliplr(input_image)), 0)

        disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
        # print(disp.squeeze().shape)
        disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)
        env = replace_pixels(5, disp_pp)

        if len(frames) == 1:
            return env
        else:
            np.save(os.path.join(output_directory, 'disparity', "{}_depth_{}.npy".format(output_name, count)), env)
            disp_to_img = scipy.misc.imresize(env.squeeze(), [original_height, original_width])
            plt.imsave(os.path.join(output_directory, 'disparity', "{}_disp_{}.png".format(output_name, count)),
                       disp_to_img, cmap='plasma')

        print("Processed Frame", count)
        count += 1

    print('done!')


def generate_depth_map_frame(params, frame, video_path):
    frames = [frame]
    return generate_depth_map_frames(params, frames, video_path)


def process_video(params, video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    frames = []
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if count % 10:
            frames.append(image)
        success, image = vidcap.read()
        count += 1
    print("Processed", count, "frames.")
    generate_depth_map_frames(params, frames, video_path)


def main(_):
    video_path = 'x.mp4'
    init()
    process_video(params, video_path)


if __name__ == '__main__':
    tf.app.run()
