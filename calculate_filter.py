from scipy import ndimage
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from isr import *
import sys

font = {'size'   : 32}
plt.rc('font', **font)

mic_num = 2
bright_idx = 0
ls_num = 6
beta = 0.01
mic = [9, 49]
w_len = 20481
test_f = w_len

def smooth(data, n=3):
    l = len(data)
    data = np.pad(data, (0, l), 'edge')
    y = np.zeros((l))
    for k in range(1, l):
        a = int(k * pow(2, -1/(2 * n)))
        b = int(k * pow(2, 1/(2 * n)))
        for i in range(a, b):
            y[k] += data[i]
        y[k] = (1 / (b - a)) * y[k]
    return y

def calculate_tf(ls_num, mic_num, type="anechoic", w_len=w_len):
    if type == "anechoic":
        prefix1 = "Anechoic Room"
        prefix2 = "AnechoicRoom_"
    elif type == "hemi":
        prefix1 = "Hemi-anechoic Room"
        prefix2 = "HemiAnechoicRoom_"
    elif type == "large":
        prefix1 = "Large Meeting Room"
        prefix2 = "LargeMeetingRoom_"
    elif type == "simulate":
        ir, tf = get_rir(ls_num, mic_num, order=0, w_len=w_len)
        return tf
    elif type == "simulate3":
        ir, tf = get_rir(ls_num, mic_num, order=3, w_len=w_len)
        return tf
    
    tf_array = np.zeros((w_len, mic_num, ls_num), complex)

    for l in range(13, 19):
        if mic_num == 2:
            for m in range(mic_num):
                name = "./" + prefix1 + "/ZoneA/PlanarMicrophoneArray/" + prefix2 + \
                    "ZoneA_PlanarMicrophoneArray_L" + str(l) + "_M" + str(mic[m]) + ".wav"
                ir, sr = librosa.load(name, sr=48000)
                ir = truncate_ir(ir, w_len=2*(w_len-1))
                tf = np.fft.rfft(ir)
                tf_array[:, m, l-13] = tf
        elif mic_num == 4:
            for m in range(mic_num):
                if m < 2:
                    name = "./" + prefix1 + "/ZoneC/PlanarMicrophoneArray/" + prefix2 + \
                        "ZoneC_PlanarMicrophoneArray_L" + str(l) + "_M" + str(mic[m]) + ".wav"
                    ir, sr = librosa.load(name, sr=48000)
                elif m >= 2:
                    name = "./" + prefix1 + "/ZoneD/PlanarMicrophoneArray/" + prefix2 + \
                        "ZoneD_PlanarMicrophoneArray_L" + str(l) + "_M" + str(mic[m-2]) + ".wav"
                    ir, sr = librosa.load(name, sr=48000)
                ir = truncate_ir(ir, w_len=w_len)
                tf = np.fft.fft(ir)
                # tf_array[:, m, l] = tf
                tf_array[:, m, l-13] = tf
    return tf_array


def calculate_ir(ls_num, mic_num, type="anechoic", w_len=w_len):
    if type == "anechoic":
        prefix1 = "Anechoic Room"
        prefix2 = "AnechoicRoom_"
    elif type == "hemi":
        prefix1 = "Hemi-anechoic Room"
        prefix2 = "HemiAnechoicRoom_"
    elif type == "large":
        prefix1 = "Large Meeting Room"
        prefix2 = "LargeMeetingRoom_"
    elif type == "simulate":
        ir, tf = get_rir(ls_num, mic_num, order=0, w_len=w_len)
        return ir
    elif type == "simulate3":
        ir, tf = get_rir(ls_num, mic_num, order=3, w_len=w_len)
        return ir
    
    # ir_array = np.zeros((w_len, mic_num, ls_num), complex)
    ir_array = np.zeros((2*(w_len-1), mic_num, ls_num), complex)

    for l in range(13, 19):
        for m in range(mic_num):
            name = "./" + prefix1 + "/ZoneA/PlanarMicrophoneArray/" + prefix2 + \
                "ZoneA_PlanarMicrophoneArray_L" + str(l) + "_M" + str(mic[m]) + ".wav"
            ir, sr = librosa.load(name, sr=48000)
            ir = truncate_ir(ir, w_len=2*(w_len-1))
            ir_array[:, m, l-13] = ir
    return ir_array


def calculate_mismatch_tf(ls_num, mic_num, type="anechoic", sample=0):
    tf = calculate_tf(ls_num, mic_num, type)
    tf_array = np.zeros((w_len, mic_num, ls_num), complex)
    for l in range(13, 19):
        for m in range(mic_num):
            ir = np.fft.ifft(tf[:, m, l-13])
            ir = np.pad(ir[0:1000], (0, len(ir)-1000))
            # ir = np.hstack((np.zeros(sample), ir))[:w_len]
            tf_array[:, m, l-13] = np.fft.fft(ir)
    return tf_array

def calculate_filter(g_a, g_b):
    ''' ACC '''
    q_matrix = np.zeros((w_len, ls_num), complex)
    for i in range(w_len):
        ga = g_a[i] 
        gb = g_b[i] 
        ga_h = ga.conj().T
        gb_h = gb.conj().T

        w, v = np.linalg.eig(np.linalg.solve(gb_h @ gb + beta * np.identity(gb.shape[1]), ga_h @ ga))
        max_eig_val = np.argmax(w)
        q = v[:, max_eig_val]
        q_matrix[i, :] = q
    return q_matrix

def calculate_contrast(ga, gb, q):
    bright = np.zeros((w_len), complex)
    dark = np.zeros((w_len), complex)
    for i in range(ls_num):
        bright += q[:, i] * ga[:, 0, i]
        # dark += q[:, i] * gb[:, 0, i]
        for j in range(mic_num - 1):
            dark += q[:, i] * gb[:, j, i]
    contrast = 20 * np.log10(np.abs(bright) / np.abs(dark))
    return contrast


def calculate_contrast_ir(ga, gb, q):
    # bright = np.zeros((22528), complex)
    # dark = np.zeros((22528), complex)
    bright = np.zeros((40960), complex)
    dark = np.zeros((40960), complex)
    for i in range(ls_num):
        q_i = np.fft.irfft(q[:, i])
        bright += np.fft.rfft(np.convolve(q_i, ga[:, 0, i]))
        dark += np.fft.rfft(np.convolve(q_i, gb[:, 0, i]))
    contrast = 20 * np.log10(np.abs(bright) / np.abs(dark))
    return contrast


def plot_q(q):
    q = np.fft.irfft(q[:, 0])
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    xx = np.arange(0, len(q)) / 48000
    axes.plot(xx, q)
    # axes.plot(xx, 20 * np.log(q))
    axes.set_xlabel('Time [s]')
    axes.set_ylabel('Amplitude [dB]')
    fig.savefig("q_linear_" + str(w_len) + ".jpg", bbox_inches="tight")


def plot_contrast(contrast, name):
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    start = int(20 * 48000 / len(contrast) / 2)
    xx = np.arange(0, len(contrast)) * 48000 / len(contrast) / 2
    # contrast = smooth(contrast)
    axes.semilogx(xx[start:], contrast[start:], linewidth=3.0)
    axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.set_xlabel('Frequency [Hz]')
    axes.set_ylabel('Amplitude [dB]')
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(name + ".jpg", bbox_inches="tight")
    np.savetxt(name + ".csv", contrast, delimiter=",")

def upsample(q):
    res_q = np.zeros((w_len, ls_num))
    for i in range(ls_num):
        q_t = np.fft.irfft(q[:, i])
        q_t = np.pad(q_t, (0, 40960 - len(q_t)))
        res_q[:, i] = np.fft.rfft(q_t)
    return res_q


for cal in ['anechoic']:
    for eval in ['anechoic', 'hemi', 'large']:
        for beta in [10e-5]:
            w_len = 20481
            test_f = w_len
            name = "./plot/0911/" + str(ls_num) + "_" + str(mic_num) + "_" + str(cal) + "_" + str(eval) + "_" + str(w_len) + "_convolve"
            tf_calculate = calculate_tf(ls_num, mic_num, cal, w_len=w_len)
            ga_calculate = tf_calculate[:, 0:1, :]
            gb_calculate = tf_calculate[:, 1:, :]
            # ga_calculate = tf_calculate[:, :, bright_idx:bright_idx+1]
            # gb_calculate = np.concatenate((tf_calculate[:, :, 0:bright_idx], tf_calculate[:, :, bright_idx+1:]), axis=2)
            q = calculate_filter(ga_calculate, gb_calculate)
            w_len = 20481
            test_f = w_len
            ir_evaluate = calculate_ir(ls_num, mic_num, eval, w_len=w_len)
            ga_evaluate = ir_evaluate[:, 0:1, :]
            gb_evaluate = ir_evaluate[:, 1:, :]
            contrast = calculate_contrast_ir(ga_evaluate, gb_evaluate, q)
            plot_contrast(contrast, name)
            sys.exit()
            # ga_evaluate = tf_evaluate[:, :, bright_idx:bright_idx+1]
            # gb_evaluate = np.concatenate((tf_evaluate[:, :, 0:bright_idx], tf_evaluate[:, :, bright_idx+1:]), axis=2)
            contrast = calculate_contrast(ga_evaluate, gb_evaluate, q)
            plot_contrast(contrast, name)
