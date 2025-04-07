import os
import numpy as np
import tgt
from scipy.stats import mode


phoneme_list = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 
                'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 
                'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 
                'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 
                'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 
                'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 
                'OW0', 'OW1', 'OW2', 'OY1', 'P', 
                'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 
                'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', 'spn']

phoneme_list = [
    "spn",
    "AA0",
    "AA1",
    "AA2",
    "AE0",
    "AE1",
    "AE2",
    "AH0",
    "AH1",
    "AH2",
    "AO0",
    "AO1",
    "AO2",
    "AW0",
    "AW1",
    "AW2",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH0",
    "EH1",
    "EH2",
    "ER0",
    "ER1",
    "ER2",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH0",
    "IH1",
    "IH2",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW0",
    "OW1",
    "OW2",
    "OY1",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH0",
    "UH1",
    "UH2",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH"
]

#pheneme_list = ["OY0", 'OY2', 'sil', 'sp']

phoneme_dict = dict()
for j, p in enumerate(phoneme_list):
    phoneme_dict[p] = j


data_dir = 'VCTK_2speakers'
mels_mode_dict = dict()
lens_dict = dict()
for p in phoneme_list:
    mels_mode_dict[p] = []
    lens_dict[p] = []
speakers = os.listdir(os.path.join(data_dir, 'mels'))
for s, speaker in enumerate(speakers):
    print('Speaker %d: %s' % (s + 1, speaker))
    textgrids = os.listdir(os.path.join(data_dir, 'textgrids', speaker))
    for textgrid in textgrids:
        t = tgt.io.read_textgrid(os.path.join(data_dir, 'textgrids', speaker, textgrid))
        m = np.load(os.path.join(data_dir, 'mels', speaker, textgrid.replace('.TextGrid', '_mel.npy')))
        t = t.get_tier_by_name('phones')
        for i in range(len(t)):
            phoneme = t[i].text
            start_frame = int(t[i].start_time * 22050.0) // 256
            end_frame = int(t[i].end_time * 22050.0) // 256 + 1
            mels_mode_dict[phoneme] += [np.round(np.median(m[:, start_frame:end_frame], 1), 1)]
            lens_dict[phoneme] += [end_frame - start_frame]

mels_mode = dict()
lens = dict()

print(1)
aaaa = 0
for p in phoneme_list:
    print(aaaa)
    print(p)
    aaaa += 1
    mels_mode[p] = mode(np.asarray(mels_mode_dict[p]), 0).mode[0]
    lens[p] = np.mean(np.asarray(lens_dict[p]))
del mels_mode_dict
del lens_dict


for s, speaker in enumerate(speakers):
    print('Speaker %d: %s' % (s + 1, speaker))
    os.mkdir(os.path.join(data_dir, 'mels_mode', speaker))
    textgrids = os.listdir(os.path.join(data_dir, 'textgrids', speaker))
    for textgrid in textgrids:
        t = tgt.io.read_textgrid(os.path.join(data_dir, 'textgrids', speaker, textgrid))
        m = np.load(os.path.join(data_dir, 'mels', speaker, textgrid.replace('.TextGrid', '_mel.npy')))
        m_mode = np.copy(m)
        t = t.get_tier_by_name('phones')
        for i in range(len(t)):
            phoneme = t[i].text
            start_frame = int(t[i].start_time * 22050.0) // 256
            end_frame = int(t[i].end_time * 22050.0) // 256 #+1
            m_mode[:, start_frame:end_frame] = np.repeat(np.expand_dims(mels_mode[phoneme], 1), end_frame - start_frame, 1)
        np.save(os.path.join(data_dir, 'mels_mode', speaker, textgrid.replace('.TextGrid', '_avgmel.npy')), m_mode)