# speech_recorder.py
# Reads the list in names.txt, and records 20 2-second clips of each name in the list
# Author: Ben Tomsett

import os
import sounddevice as sd
import soundfile as sf

os.system("clear")

file = open("names.txt", "r")
names = file.read().split("\n")
file.close()

print("Loaded " + str(len(names)) + " names.")
input("Press enter to start...")

# Recording constants
sample_rate = 16000
clip_length = 2
num_recordings = 10

for name in names:
    i = 0

    while i < num_recordings:
        os.system("clear")

        print("Name: " + name)
        print("Recording " + str(i + 1) + " of " + str(num_recordings))

        input("Press enter to start recording...")

        audio = sd.rec(clip_length * sample_rate, channels=1, samplerate=sample_rate)
        sd.wait()

        print("Recording finished, playing back audio...")

        sd.play(audio, sample_rate)
        sd.wait()

        save = input("Save audio? (y/n): ").lower()

        if save == "y":
            print("Recording finished, saving .wav file...")
            sf.write(f'./audio/testing/{name}-{i + 1}.wav', audio, sample_rate)
            print("Saved .wav file.")

            i += 1
        else:
            print("Recording discarded.")

# Record audio and wait until finished recording
# print("Starting recording...")
# r = sd.rec(seconds * fs, channels=1, samplerate=fs)
# sd.wait()
#
# print("Recording finished, plotting waveform and starting playback")
#
# print("Saving .wav file...")
# sf.write(f'speech{round(time.time() * 1000)}.wav', r, fs)



