import pandas as pd 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np


font = {'size'   : 32}
plt.rc('font', **font)

folder = "./plot/"
test_f = 20480

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
    return y


'''calc: anechoic; eval: 3 rooms; 6 2'''
def plot1():
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    for (calc, eval) in (["anechoic", "anechoic"], ["anechoic", "hemi"], ["anechoic", "large"]):
        name = "0911/6_2_" + str(calc) + "_" + str(eval) + "_20481.csv"
        # name = "contrast_6_2_" + str(calc) + "_" + str(eval) + ".csv"
        contrast = pd.read_csv(folder + name).values[:, 0]
        start = int(20 * 48000 / len(contrast) / 2)
        xx = np.arange(0, test_f) * 48000 / test_f / 2
        axes.semilogx(xx[start:], smooth(contrast)[start:], label = str(calc) + "_" + str(eval), linewidth=3.0, ls="--")
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.set_xlabel('Frequency [Hz]')
    axes.set_ylabel('Contrast [dB]')
    axes.set_title("6_2")
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(folder + "plot1.jpg", bbox_inches="tight")

'''calc: 3; eval: large'''
def plot2():
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    for (calc, eval) in (["anechoic", "large"], ["hemi", "large"], ["large", "large"]):
        name = "contrast_6_2_" + str(calc) + "_" + str(eval) + ".csv"
        contrast = pd.read_csv(folder + name).values[:, 0]
        xx = np.arange(1, test_f) * 48000 // test_f//2
        axes.semilogx(xx, smooth(contrast), label = str(calc) + "_" + str(eval), linewidth=3.0, ls="--")
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.set_xlabel('Frequency [Hz]')
    axes.set_ylabel('Contrast [dB]')
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(folder + "plot2.jpg", bbox_inches="tight")


'''calc = eval'''
def plot3():
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    for (calc, eval) in (["anechoic", "anechoic"], ["hemi", "hemi"], ["large", "large"]):
        name = "contrast_6_2_" + str(calc) + "_" + str(eval) + ".csv"
        contrast = pd.read_csv(folder + name).values[:, 0]
        xx = np.arange(1, test_f) * 48000 // test_f//2
        axes.semilogx(xx, smooth(contrast), label = str(calc) + "_" + str(eval), linewidth=3.0, ls="--")
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.set_xlabel('Frequency [Hz]')
    axes.set_ylabel('Contrast [dB]')
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(folder + "plot3.jpg", bbox_inches="tight")


'''anechoic, 30, 6, 4, 2'''
def plot4():
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    for ls in [30, 6]:
        for mic in [4, 2]:
            name = "contrast_" + str(ls) + "_" + str(mic) + "_anechoic_anechoic.csv"
            contrast = pd.read_csv(folder + name).values[:, 0]
            xx = np.arange(1, test_f) * 48000 // test_f//2
            axes.semilogx(xx, smooth(contrast), label = str(ls) + "_" + str(mic), linewidth=3.0, ls="--")
            axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
            axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.set_xlabel('Frequency [Hz]')
    axes.set_ylabel('Contrast [dB]')
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(folder + "plot4.jpg", bbox_inches="tight")

''' simulate, anechoic, 0, 3 '''
def plot5():
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    for (calc, eval) in (["simulate", "anechoic"], ["simulate3", "anechoic"]):
        name = "contrast_6_2_" + str(calc) + "_" + str(eval) + ".csv"
        contrast = pd.read_csv(folder + name).values[:, 0]
        xx = np.arange(1, test_f) * 48000 // test_f//2
        axes.semilogx(xx, smooth(contrast), label = str(calc) + "_" + str(eval), linewidth=3.0, ls="--")
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.set_xlabel('Frequency [Hz]')
    axes.set_ylabel('Contrast [dB]')
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(folder + "plot5.jpg", bbox_inches="tight")

''' mismatch '''
def plot6():
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    for sample in [0, 1, 5, 10, 20, 100]:
        name = "mismatch_6_2_" + str(sample) + ".csv"
        contrast = pd.read_csv(folder + name).values[:, 0]
        xx = np.arange(1, test_f) * 48000 // test_f//2
        axes.semilogx(xx, smooth(contrast), label = str(sample), linewidth=3.0, ls="--")
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.set_xlabel('Frequency [Hz]')
    axes.set_ylabel('Contrast [dB]')
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(folder + "plot6.jpg", bbox_inches="tight")

''' line '''
def plot7():
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    xx = [0, 1, 5, 10, 20, 100]
    y = np.zeros((len(xx)))
    for i in range(len(xx)):
        name = "mismatch_6_2_" + str(xx[i]) + ".csv"
        contrast = pd.read_csv(folder + name).values[:, 0]
        y[i] = np.mean(contrast)
    axes.plot(xx, y, linewidth=3.0)
    axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.set_xlabel('Samples')
    axes.set_ylabel('Average Contrast [dB]')
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(folder + "plot7.jpg", bbox_inches="tight")


'''calc: anechoic; eval: 3 rooms; 6 4'''
def plot8():
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    for (calc, eval) in (["anechoic", "anechoic"], ["anechoic", "hemi"], ["anechoic", "large"]):
        name = "contrast_6_4_" + str(calc) + "_" + str(eval) + ".csv"
        contrast = pd.read_csv(folder + name).values[:, 0]
        xx = np.arange(1, test_f) * 48000 // test_f//2
        axes.semilogx(xx, smooth(contrast), label = str(calc) + "_" + str(eval), linewidth=3.0, ls="--")
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.set_xlabel('Frequency [Hz]')
    axes.set_ylabel('Contrast [dB]')
    axes.set_title("6_4")
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(folder + "plot8.jpg", bbox_inches="tight")

'''calc: anechoic; eval: 3 rooms; 30 4'''
def plot9():
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    for (calc, eval) in (["anechoic", "anechoic"], ["anechoic", "hemi"], ["anechoic", "large"]):
        name = "contrast_30_4_" + str(calc) + "_" + str(eval) + ".csv"
        contrast = pd.read_csv(folder + name).values[:, 0]
        xx = np.arange(1, test_f) * 48000 // test_f//2
        axes.semilogx(xx, smooth(contrast), label = str(calc) + "_" + str(eval), linewidth=3.0, ls="--")
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.set_xlabel('Frequency [Hz]')
    axes.set_ylabel('Contrast [dB]')
    axes.set_title("30_4")
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(folder + "plot9.jpg", bbox_inches="tight")


'''calc: simulate; eval: 3 rooms; 6 2'''
def plot10():
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    for (calc, eval) in (["simulate", "anechoic"], ["simulate", "hemi"], ["simulate", "large"]):
        name = "0905_40960/6_2_" + str(calc) + "_" + str(eval) + "_0.0001.csv"
        # name = "contrast_6_2_" + str(calc) + "_" + str(eval) + ".csv"
        contrast = pd.read_csv(folder + name).values[:, 0]
        start = int(20 * 48000 / len(contrast) / 2)
        xx = np.arange(0, test_f) * 48000 // test_f//2
        axes.semilogx(xx[start:], smooth(contrast)[start:], label = str(calc) + "_" + str(eval), linewidth=3.0, ls="--")
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.set_xlabel('Frequency [Hz]')
    axes.set_ylabel('Contrast [dB]')
    axes.set_title("6_2")
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(folder + "plot10.jpg", bbox_inches="tight")

def plot11():
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    for beta in [0, 1, 0.1, 0.01, 10e-5]:
        name = "mismatch_contrast_6_2_simulate_large_" + str(beta) + ".csv"
        contrast = pd.read_csv(folder + name).values[:, 0]
        xx = np.arange(1, test_f) * 48000 // test_f//2
        axes.semilogx(xx, smooth(contrast), label = str(beta), linewidth=3.0, ls="--")
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.set_xlabel('Frequency [Hz]')
    axes.set_ylabel('Contrast [dB]')
    axes.set_title("6_2 simulate large")
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(folder + "plot11.jpg", bbox_inches="tight")


'''calc: simulate; eval: 3 rooms; 6 2'''
def plot12():
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    for (calc, eval) in (["simulate", "anechoic"], ["simulate", "hemi"], ["simulate", "large"]):
        name = "mismatch_contrast_6_2_" + str(calc) + "_" + str(eval) + ".csv"
        contrast = pd.read_csv(folder + name).values[:, 0]
        xx = np.arange(1, test_f) * 48000 // test_f//2
        axes.semilogx(xx, smooth(contrast), label = str(calc) + "_" + str(eval), linewidth=3.0, ls="--")
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.set_xlabel('Frequency [Hz]')
    axes.set_ylabel('Contrast [dB]')
    axes.set_title("6_2")
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(folder + "plot12.jpg", bbox_inches="tight")



'''todo simulate real'''
def plot13():
    folder = "./plot/0828_simulate_real/"
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    for (calc, eval) in (["simulate", "anechoic"], ["anechoic", "anechoic"]):
        name = "6_2_" + str(calc) + "_" + str(eval) + ".csv"
        contrast = pd.read_csv(folder + name).values[:, 0]
        xx = np.arange(1, test_f) * 48000 // test_f//2
        axes.semilogx(xx, smooth(contrast), label = str(calc) + "_" + str(eval), linewidth=3.0, ls="--")
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.set_xlabel('Frequency [Hz]')
    axes.set_ylabel('Contrast [dB]')
    axes.set_title("6_2")
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(folder + "plot13.jpg", bbox_inches="tight")


'''calc: simulate3; eval: 3 rooms; 6 2'''
def plot14():
    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(111)
    for (calc, eval) in (["simulate3", "anechoic"], ["simulate3", "hemi"], ["simulate3", "large"]):
        name = "0905_40960/6_2_" + str(calc) + "_" + str(eval) + "_0.0001.csv"
        # name = "contrast_6_2_" + str(calc) + "_" + str(eval) + ".csv"
        contrast = pd.read_csv(folder + name).values[:, 0]
        start = int(20 * 48000 / len(contrast) / 2)
        xx = np.arange(0, test_f) * 48000 // test_f//2
        axes.semilogx(xx[start:], smooth(contrast)[start:], label = str(calc) + "_" + str(eval), linewidth=3.0, ls="--")
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axes.set_xlabel('Frequency [Hz]')
    axes.set_ylabel('Contrast [dB]')
    axes.set_title("6_2")
    leg = axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    for line in leg.get_lines():
        line.set_linewidth(4)
    fig.savefig(folder + "plot14.jpg", bbox_inches="tight")


plot1()

