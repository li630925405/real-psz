import pandas as pd 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np
from scipy import ndimage
from scipy import signal
import librosa
from isr import *

font = {'size'   : 32}
plt.rc('font', **font)

# 1/3 octave smoothing
def smooth(data, n=3):
    data = np.pad(data, (0, test_f), 'edge')
    y = np.zeros((test_f))
    for k in range(1, test_f):
        a = int(k * pow(2, -1/(2 * n)))
        b = int(k * pow(2, 1/(2 * n)))
        for i in range(a, b):
            y[k] += data[i]
        y[k] = (1 / (b - a)) * y[k]
    return y[1:test_f]

def mic_pos(m):
    xm = -0.14 + 0.04 * ((m - 1) % 8) - 0.4
    ym = 0.14 - 0.04 * ((m - 1) // 8)
    return round(xm, 2), round(ym, 2)

def ls_pos(l):
    R = 1.5 - 0.0008 * 340
    x = -R * np.sin((2 * l - 1) * np.pi / 60)
    y = R * np.cos((2 * l - 1) * np.pi / 60)
    return round(x, 2), round(y, 2)

def print_pos(x, y):
    return "(x: " + str(x) + " y: " + str(y) + ")"

def print_mic(m):
    x, y = mic_pos(m)
    return "mic " + str(m) + " " + print_pos(x, y)

def print_ls(l):
    x, y = ls_pos(l)
    return "ls " + str(l) + " " + print_pos(x, y)

def ir2fr(ir):
    return 20 * np.log10(np.abs(np.fft.rfft(ir))) 

def truncate_ir(ir, w_len=1024):
    window = signal.windows.tukey(w_len, alpha=0.005)
    ir = ir[:w_len] * window
    return ir

def plot(data, labels, name, type, title="", sr=48000, ylim=None):
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    for i in range(len(data)):
        if type == "ir":
            xx = np.arange(len(data[i])) / sr
            axes.plot(xx, data[i], linewidth=3, label=labels[i])
            # axes.plot(xx, 20 * np.log10(np.abs(data[i])), linewidth=3, label=labels[i], ls="-")
        elif type == "fr":
            xx = np.arange(1, test_f) * sr / test_f/2
            axes.semilogx(xx, smooth(data[i]), label=labels[i], linewidth=3.0, ls=":")
            # axes.semilogx(xx, data[i][1:], label=labels[i], linewidth=3.0, ls=":")
    if type == "ir":
        axes.set_xlabel('Time [s]')
        axes.set_ylabel('Amplitude')
    elif type == "fr":
        axes.set_xlabel('Frequency [Hz]')
        axes.set_ylabel('Amplitude [dB]')
    if ylim != None:
        axes.set_ylim(ylim)
    axes.set_title(title)
    axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(name, bbox_inches="tight") 


def plot_fr_linear(data, labels, name, title="", sr=48000, ylim=None):
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    for i in range(len(data)):
        xx = np.arange(1, test_f) * sr // test_f//2
        axes.plot(xx, smooth(data[i]), label=labels[i], linewidth=3.0, ls=":")
    axes.set_xlabel('Frequency [Hz]')
    axes.set_ylabel('Amplitude [dB]')
    if ylim != None:
        axes.set_ylim(ylim)
    axes.set_title(title)
    axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(name, bbox_inches="tight") 


def plot_pos(l, m1, m2, type):
    ir1, sr = librosa.load(folder + "AnechoicRoom_ZoneA_PlanarMicrophoneArray_L" + str(l) + "_M" + str(m1) + ".wav", sr=48000)
    ir2, sr = librosa.load(folder + "AnechoicRoom_ZoneA_PlanarMicrophoneArray_L" + str(l) + "_M" + str(m2) + ".wav", sr=48000)
    ir1 = truncate_ir(ir1)
    ir2 = truncate_ir(ir2)
    labels = [print_mic(m1), print_mic(m2)]

    if type == "ir":
        data = [ir1, ir2]
        name = "./plot/ir" + str(l) + str(m1) + str(m2) + ".jpg"
        plot(data, labels, name, "ir", print_ls(l))
    elif type == "fr":
        fr1 = ir2fr(ir1)
        fr2 = ir2fr(ir2)
        data = [fr1, fr2]
        name = "./plot/fr" + str(l) + str(m1) + str(m2) + ".jpg"
        plot(data, labels, name, "fr", print_ls(l))

def plot_pos_1(l, m1, type):
    ir1, sr = librosa.load(folder + "AnechoicRoom_ZoneA_PlanarMicrophoneArray_L" + str(l) + "_M" + str(m1) + ".wav", sr=48000)
    ir1 = truncate_ir(ir1, w_len=512)
    labels = [print_mic(m1)]
    x_m, y_m = mic_pos(m1)
    x_l, y_l = ls_pos(l)
    print("distance: ", np.sqrt(np.sum((np.array([x_m, y_m]) - np.array([x_l, y_l])) ** 2)))

    if type == "ir":
        data = [ir1]
        name = "./plot/ir" + str(l) + str(m1) + ".jpg"
        plot(data, labels, name, "ir", print_ls(l))
        np.savetxt("./ir_" +  str(l) + str(m1) + ".csv", ir1)
    elif type == "fr":
        fr1 = ir2fr(ir1)
        data = [fr1]
        name = "./plot/fr" + str(l) + str(m1) + ".jpg"
        plot(data, labels, name, "fr", print_ls(l))


def plot_room(type):
    ir, fr = get_rir(6, 2, w_len=1024, order=3, absor=0.7, type="large")
    ir0 = ir[:, 0, 0]
    ir0 = np.pad(ir0, (0, w_len-len(ir0)))
    ir1, sr = librosa.load("./Anechoic Room/ZoneA/PlanarMicrophoneArray/AnechoicRoom_ZoneA_PlanarMicrophoneArray_L1_M1.wav", sr=48000)
    ir2, sr = librosa.load("./Hemi-anechoic Room/ZoneA/PlanarMicrophoneArray/HemiAnechoicRoom_ZoneA_PlanarMicrophoneArray_L1_M1.wav", sr=48000)
    ir3, sr = librosa.load("./Large Meeting Room/ZoneA/PlanarMicrophoneArray/LargeMeetingRoom_ZoneA_PlanarMicrophoneArray_L1_M1.wav", sr=48000)
    
    ir1 = truncate_ir(ir1, w_len=w_len)
    ir2 = truncate_ir(ir2, w_len=w_len)
    # ir3 = truncate_ir(ir3, w_len=4096)
    ir3 = truncate_ir(ir3, w_len=w_len)
    labels = ["large", "hemi", "anechoic", "sim"]

    if type == "ir":
        data = [ir3, ir2, ir1, ir0]
        name = "./plot/sim_ir_linear_" + str(w_len) + ".jpg"
        plot(data, labels, name, "ir")
    elif type == "fr":
        fr1 = ir2fr(ir1)
        fr2 = ir2fr(ir2)
        fr3 = ir2fr(ir3)
        data = [fr3, fr2, fr1]
        name = "./plot/room_fr_" + str(w_len) + ".jpg"
        plot(data, labels, name, "fr")


def plot_room_ir(type):
    if type == "anechoic":
        ir, sr = librosa.load("./Anechoic Room/ZoneA/PlanarMicrophoneArray/AnechoicRoom_ZoneA_PlanarMicrophoneArray_L1_M1.wav", sr=48000)
    elif type == "hemi":
        ir, sr = librosa.load("./Hemi-anechoic Room/ZoneA/PlanarMicrophoneArray/HemiAnechoicRoom_ZoneA_PlanarMicrophoneArray_L1_M1.wav", sr=48000)
    elif type == "large":
        ir, sr = librosa.load("./Large Meeting Room/ZoneA/PlanarMicrophoneArray/LargeMeetingRoom_ZoneA_PlanarMicrophoneArray_L1_M1.wav", sr=48000)
    ir = truncate_ir(ir, w_len=4096)
    labels = [type]

    data = [ir]
    name = "./plot/ir_" + type + "_4096.jpg"
    plot(data, labels, name, "ir", ylim=[-0.3, 0.3])


def plot_sim(room="hemi"):
    ir, fr = get_rir(6, 2, w_len=1024, order=3, absor=0.7, type=room)
    ir1 = ir[:, 0, 0]
    ir2, sr = librosa.load("./Anechoic Room/ZoneA/PlanarMicrophoneArray/AnechoicRoom_ZoneA_PlanarMicrophoneArray_L13_M9.wav", sr=48000)
    ir1 = truncate_ir(ir1)
    ir2 = truncate_ir(ir2)
    labels = ["simulate", "real"]
    data = [ir1, ir2]
    name = "./plot/sim_ir.jpg"
    plot(data, labels, name, "ir")
    np.savetxt("./ir_csv/sim_ir_" +  str(l) + str(m1) + ".csv", ir1)
    np.savetxt("./ir_csv/real_ir_" +  str(l) + str(m1) + ".csv", ir2)

def plot_origin():
    ir, sr = librosa.load("./Anechoic Room/ZoneA/PlanarMicrophoneArray/AnechoicRoom_ZoneA_PlanarMicrophoneArray_L13_M9.wav", sr=48000)
    ir = truncate_ir(ir)
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    xx = np.arange(len(ir)) / sr
    axes.plot(xx, ir, linewidth=3, ls=":")
    axes.set_xlabel('Time [s]')
    axes.set_ylabel('Amplitude')
    name = "tmp.png"
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(name, bbox_inches="tight") 

folder = "./Anechoic Room/ZoneA/PlanarMicrophoneArray/"

m1 = 9
m2 = 60
l = 13
w_len = 4096
test_f = w_len

plot_room("ir")