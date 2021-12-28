import soundfile
import sounddevice
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display


# Constants
third_octave_boundries = np.array(
    [
        11.2,
        14.1,
        17.8,
        22.4,
        28.2,
        35.5,
        44.7,
        56.2,
        70.8,
        89.1,
        112,
        141,
        178,
        224,
        282,
        355,
        447,
        562,
        708,
        891,
        1122,
        1413,
        1778,
        2239,
        2818,
        3548,
        4467,
        5623,
        7079,
        8913,
        11220,
        14130,
        17780,
        22390,
    ]
)
third_octave_center = np.array(
    [
        12.5,
        16,
        20,
        25,
        31.5,
        40,
        50,
        63,
        80,
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
        5000,
        6300,
        8000,
        10000,
        12500,
        16000,
        20000,
    ]
)
third_octave_names = [
    "12.5",
    "16",
    "20",
    "25",
    "31.5",
    "40",
    "50",
    "63",
    "80",
    "100",
    "125",
    "160",
    "200",
    "250",
    "315",
    "400",
    "500",
    "630",
    "800",
    "1k",
    "1.25k",
    "1.6k",
    "2k",
    "2.5k",
    "3.15k",
    "4k",
    "5k",
    "6.3k",
    "8k",
    "10k",
    "12.5k",
    "16k",
    "20k",
]

# Input, Output
def search_device(name):
    devicelist = sounddevice.query_devices()
    d = 0
    for device in devicelist:
        if name in device["name"]:
            return d
        d = d + 1
    print("Device not found!")
    return sounddevice.default.device


def print_devicelist():
    print(sounddevice.query_devices())


def play_track(m, sr, device=sounddevice.default.device, channels=None, blocking=True):
    sounddevice.play(
        m, samplerate=sr, mapping=channels, blocking=blocking, device=device
    )


def rec_track(
    time, channels, sr=48000, device=sounddevice.default.device, blocking=True
):
    m = sounddevice.rec(
        frames=int(time * sr),
        samplerate=sr,
        device=device,
        mapping=channels,
        dtype=np.double,
        blocking=blocking,
    )
    return m, sr


def play_rec_track(
    m, sr, channels_out, channels_in, device=sounddevice.default.device, blocking=True
):
    n = sounddevice.playrec(
        m,
        samplerate=sr,
        device=device,
        input_mapping=channels_in,
        output_mapping=channels_out,
        blocking=blocking,
    )
    return n, sr


def load_track(path):
    m, sr = librosa.load(
        path, sr=48000, mono=True, dtype=np.double, res_type="kaiser_best"
    )
    m = normalize_headroom(m)
    return m, sr


def normalize_headroom(m: np.array, peak_value=0.75) -> np.array:
    curr_peak_value = np.amax(m)
    if np.abs(curr_peak_value - peak_value) > 0.01:
        factor = peak_value / curr_peak_value
        m = m * factor
    return m


def norm(m: np.array) -> np.array:
    return normalize_headroom(m)


def save_track(path, m, sr=48000):
    soundfile.write(path, m, sr, subtype="PCM_24", format="WAV")


def pad(m, sr, pre_delay=0.0, post_delay=0.0):
    return np.pad(m, (int(pre_delay * sr), int(post_delay * sr)), constant_values=0.0)


# Visualization


def add_wave_plot(m, sr):
    librosa.display.waveshow(m, sr=sr)


def add_two_wave_plot(m1, m2, sr, overlap=False):
    if overlap:
        fig, ax = plt.subplots(nrows=1, sharex=True)
        librosa.display.waveshow(m1, sr=sr, alpha=0.5, ax=ax, label="First")
        librosa.display.waveshow(m2, sr=sr, alpha=0.5, color="r", ax=ax, label="Second")
    else:
        fig, ax = plt.subplots(nrows=2, sharex=True)
        librosa.display.waveshow(m1, sr=sr, alpha=0.5, ax=ax[0])
        librosa.display.waveshow(m1, sr=sr, alpha=0.5, ax=ax[1])


def add_hist(m, sr=48000):
    D = librosa.stft(m)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=1.0)
    plt.figure()
    librosa.display.specshow(S_db, x_axis="time", y_axis="log", sr=sr)
    plt.colorbar()


def add_spec(m, sr=48000):
    plt.magnitude_spectrum(m, Fs=sr, scale="dB")


def add_phase(m, sr=48000):
    plt.phase_spectrum(m, Fs=sr)


def add_spec_bands(m, sr=48000):
    amps, freqs, _ = plt.magnitude_spectrum(m, Fs=sr)
    plt.clf()
    cuts = np.searchsorted(freqs, third_octave_boundries[1:-1])
    bin_amps = np.zeros(len(third_octave_center))
    bin_amps[0] = np.sum(amps[0 : cuts[0]])
    bin_amps[-1] = np.sum(amps[cuts[-1] :])
    for i in range(1, len(cuts)):
        bin_amps[i] = np.sum(amps[cuts[i - 1] : cuts[i]])

    bin_amps = np.clip(10 * np.log10(bin_amps) + 30.0, 0.0, None)
    plt.bar(third_octave_names, bin_amps)
    plt.xlabel("Freq [Hz]")
    plt.ylabel("Amplitude")


def calc_xth_octave_band_cuts(
    x: np.double, fmin: np.double, fmax: np.double
) -> np.ndarray:
    factor = np.power(2.0, 1.0 / x)
    i = 0
    cuts = [fmin]
    while cuts[i] * factor < fmax:
        cuts.append(cuts[i] * factor)
        i = i + 1
    return np.array(cuts)


def calc_xth_octave_band_names(cuts):
    out = ["low"]
    for i in range(1, len(cuts)):
        out.append("{:.2f}".format((cuts[i] + cuts[i - 1]) / 2.0))
    out.append("high")
    return out


def add_spec_xbands(m, sr=48000, x=6.0):
    amps, freqs, _ = plt.magnitude_spectrum(m, Fs=sr)
    plt.clf()
    xth_bound = calc_xth_octave_band_cuts(x, 12.5, 20000)
    cuts = np.searchsorted(freqs, xth_bound)
    bin_amps = np.zeros(len(xth_bound) + 1)
    bin_amps[0] = np.sum(amps[0 : cuts[0]])
    bin_amps[-1] = np.sum(amps[cuts[-1] :])
    for i in range(1, len(cuts)):
        bin_amps[i] = np.sum(amps[cuts[i - 1] : cuts[i]])

    bin_amps = np.clip(10 * np.log10(bin_amps) + 30.0, 0.0, None)
    plt.bar(calc_xth_octave_band_names(xth_bound), bin_amps)
    plt.xticks(rotation=90)
    plt.xlabel("Freq [Hz]")
    plt.ylabel("Amplitude")


def main():
    print_devicelist()
    RUBIX44 = search_device("Rubix44")
    m1, sr = load_track("hallo.wav")
    m2, sr = load_track("testfull.wav")
    add_spec_xbands(m1, x=12.0)
    plt.show()
    # save_track('hallo.wav', m, sr)


if __name__ == "__main__":
    main()
