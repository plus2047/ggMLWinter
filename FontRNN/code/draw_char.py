#!/usr/bin/env python
# coding: utf-8

import os
import random
import numpy as np
# from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

import train
import model
import utils
import gmm

data_dir =  '../FontRNN/data'
model_dir = '../FontRNN/log/SXmodel1030-1013'

# load dataset and paramters
[train_set, valid_set, test_set, std_train_set, std_valid_set, std_test_set, 
     hps_model, eval_hps_model] = train.load_env(data_dir, model_dir)

train_size, valid_size, test_size = 2000, 1000, 731

# construct model:
train.reset_graph()
train_model = model.FontRNN(hps_model)
eval_model = model.FontRNN(eval_hps_model, reuse=True)

# load trained checkpoint
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
train.load_checkpoint(sess, model_dir)

data = np.load("../data/1030-1013-tag-only.npz")
tag_list = data["tag"]
word2idx = {w : i for i, w in enumerate(tag_list)}


def test_model(sess, testmodel, input_stroke):
    stroke_len = len(input_stroke)
    input_stroke = utils.to_big_strokes(input_stroke, max_len=testmodel.hps.max_seq_len).tolist()
    input_stroke.insert(0, [0, 0, 1, 0, 0])
    feed = {testmodel.enc_input_data: [input_stroke],
            testmodel.enc_seq_lens: [stroke_len],
            }
    output = sess.run([testmodel.pi, testmodel.mu1, testmodel.mu2, testmodel.sigma1,
                       testmodel.sigma2, testmodel.corr, testmodel.pen,
                       testmodel.timemajor_alignment_history],
                      feed)
    gmm_params = output[:-1]
    timemajor_alignment_history = output[7]

    return gmm_params, timemajor_alignment_history

def sample_from_params(params, temp=0.1, greedy=False):
    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen] = params

    max_len = o_pi.shape[0]
    num_mixture = o_pi.shape[1]

    strokes = np.zeros((max_len, 5), dtype=np.float32)

    for step in range(max_len):
        next_x1 = 0
        next_x2 = 0
        eos = [0, 0, 0]
        eos[np.argmax(o_pen[step])] = 1
        for mixture in range(num_mixture):
            x1, x2 = gmm.sample_gaussian_2d(o_mu1[step][mixture], o_mu2[step][mixture],
                                            o_sigma1[step][mixture], o_sigma2[step][mixture],
                                            o_corr[step][mixture], np.sqrt(temp), greedy)
            next_x1 += x1 * o_pi[step][mixture]
            next_x2 += x2 * o_pi[step][mixture]
        strokes[step, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]
    strokes = utils.to_normal_strokes(strokes)
    return strokes

def to_absolute_coordinate(mat, scale_factor=300):
    low_tri_matrix = np.tril(np.ones((mat.shape[0], mat.shape[0])), 0)
    mat[:, :2] = np.rint(scale_factor * np.matmul(low_tri_matrix, mat[:, :2]))
    return mat

def sub_draw(mat, count_lim):
    cnt = pre_i = 0
    xmax, xmin = np.max(mat[:, 0]), np.min(mat[:, 0])
    ymax, ymin = np.max(mat[:, 1]), np.min(mat[:, 1])
    for i in range(mat.shape[0]):
        if mat[i][2] == 1:
            plt.plot(
                mat[pre_i:i + 1, 0],
                mat[pre_i:i + 1, 1],
                linewidth=3)
            cnt += 1
            pre_i = i + 1
            if cnt >= count_lim:
                break
    # plt.axis('off')
    # plt.hlines(ylim / 2, 0, xlim)
    # plt.vlines(xlim / 2, 0, ylim)
    plt.xlim((xmin - 10, xmax + 10))
    plt.ylim((ymin - 10, ymax + 10))
    plt.xticks([])
    plt.yticks([])
    plt.gca().invert_yaxis()
    # plt.title("Groud Truth")

def draw_grid(mlist, draw_count_lim, width=16):
    def to_abs_list(mat_list):
        return [to_absolute_coordinate(mat.copy()) if mat is not None else None for mat in mat_list]

    mlist = to_abs_list(mlist)
    count = len(mlist)
    lines = (count - 1) // width + 1
    
    plt.figure(figsize=(3 * width, 3 * lines))
    for i in range(count):
        plt.subplot(lines, width, 1 + i)
        if mlist[i] is not None:
            sub_draw(mlist[i], draw_count_lim)

def index2mat(index):
    if index < train_size:
        std_dataset = std_train_set
        dataset = train_set
    elif index < train_size + valid_size:
        std_dataset = std_valid_set
        dataset = valid_set
        index -= train_size
    else:
        std_dataset = std_test_set
        dataset = test_set
        index -= train_size + valid_size

    from_strokes = np.copy(std_dataset.strokes[index])
    to_strokes = np.copy(dataset.strokes[index])
    params, timemajor_alignment_history = test_model(sess, eval_model, from_strokes)
    generated_strokes = sample_from_params(params, greedy=True)

    return from_strokes, to_strokes, generated_strokes

def draw_by_index_list(index_list):
    mlist = []
    for index in index_list:
        if index != -1:
            from_strokes, to_strokes, generated_strokes = index2mat(index)
            mlist.append(generated_strokes)
        else:
            mlist.append(None)
    draw_grid(mlist, 10000)


def save_fig(sentence, filename="static/generated.png"):
    idx_list = [word2idx[w] if w in word2idx else -1 for w in sentence]
    draw_by_index_list(idx_list)
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    s = "大江东去浪淘尽千古风流人物"
    save_fig(s)
