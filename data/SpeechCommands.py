import os
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import yaml

torchaudio.set_audio_backend("sox_io")

SPEECH_COMMANDS_SHAPE = [1, 16000]
SPEECH_COMMANDS_N_CLASSES = 35
SPEECH_COMMANDS_N_ENVS = 2618


class SpeechCommands(SPEECHCOMMANDS):
    def __init__(self, root, lookup_filename, split_filename, split, download=True, pad_to=16000):
        os.makedirs(root, exist_ok=True)
        super(SpeechCommands, self).__init__(root=root, url='speech_commands_v0.02',
                                             folder_in_archive='SpeechCommands', download=download)

        assert split in {'train', 'test', 'valid', 'train+valid'}

        self.pad_to = pad_to

        with open(lookup_filename) as lookup_file:
            self.lookup = yaml.safe_load(lookup_file)
        with open(split_filename) as split_file:
            split_ids = yaml.safe_load(split_file)

        if split == 'train+valid':
            self.id_lookup = split_ids['train'].update(split_ids['valid'])
        else:
            self.id_lookup = split_ids[split]

    def __getitem__(self, index):
        index = self.id_lookup[index]
        waveform, sample_rate, label, speaker_id, utterance_number = super(SpeechCommands, self).__getitem__(index)
        pad_len = self.pad_to - waveform.shape[1]
        if pad_len > 0:
            pad_signal = torch.zeros(1, pad_len) + waveform[0, 0]
            waveform = torch.cat([pad_signal, waveform], 1)

        label = self.lookup['label'][label]
        sid = self.lookup['sid'][speaker_id]

        return {'x': waveform, 'y': torch.LongTensor([label]), 'e': torch.LongTensor([sid])}

    def __len__(self):
        return len(self.id_lookup)
