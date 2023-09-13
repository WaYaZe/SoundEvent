from pydub import AudioSegment

sound = AudioSegment.from_wav("./test3.wav")

one_seconds = 3 * 1000
part = 0

for part in range(2500):
    head = part * one_seconds
    tail = (part + 1) * one_seconds
    sound_part = sound[head:tail]
    name = './testaudio/v3_' + str(part) + '.wav'
    sound_part.export(name, format='wav')
