from tqdm import tqdm
import utils, time
import numpy as np
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from model import train_audio_transforms

from scipy.signal import medfilt

def predict_align(args, model, test_data, device, model_type):

    if not os.path.exists(args.pred_dir):
        os.makedirs(args.pred_dir)

    dataloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=0,
                                             collate_fn=utils.my_collate)
    model.eval()

    resolution = 256 / 22050 * 3

    with tqdm(total=len(test_data)) as pbar, torch.no_grad():
        for batch_idx, _data in enumerate(dataloader):
            spectrograms, phones, input_lengths, phone_lengths, _ = _data
            spectrograms, phones = spectrograms.to(device), phones.to(device)

            # predict
            all_outputs = model(spectrograms)
            all_outputs = F.log_softmax(all_outputs, dim=2)

            batch_num, output_length, num_classes = all_outputs.shape

            song_pred = all_outputs.data.cpu().numpy().reshape(-1, num_classes) # total_length, num_classes
            total_length = int(audio_length / args.sr // resolution)
            song_pred = song_pred[:total_length, :]

            # smoothing
            P_noise = np.random.uniform(low=1e-11, high=1e-10, size=song_pred.shape)
            song_pred = np.log(np.exp(song_pred) + P_noise)

            # dynamic programming alignment
            word_align, score = utils.alignment(song_pred, words, idx)
            print("\t{}:\t{}".format(audio_name, score))

            # write
            with open(os.path.join(args.pred_dir, audio_name + "_align.csv"), 'w') as f:
                for j in range(len(word_align)):
                    word = word_align[j]
                    f.write("{},{},{}\n".format(word[0] * resolution, word[1] * resolution, words[idx[j,0]:idx[j,1]]))

            pbar.update(1)

    return 0


def validate(batch_size, model, target_frame, criterion, dataloader, device, model_type, loss_w=0.1):

    avg_time = 0.
    model.eval()

    total_loss = 0.
    data_len = len(dataloader.dataset) // batch_size

    with tqdm(total=data_len) as pbar:
        for batch_idx, _data in enumerate(dataloader):
            spectrograms, phones, input_lengths, phone_lengths, _ = _data
            spectrograms, phones = spectrograms.to(device), phones.to(device)

            t = time.time()

            output_phone = model(spectrograms)  # (batch, time, n_class)
            output_phone = F.log_softmax(output_phone, dim=2)
            output_phone = output_phone.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output_phone, phones, input_lengths, phone_lengths)

            t = time.time() - t
            avg_time += (1. / float(batch_idx + 1)) * (t - avg_time)

            total_loss += loss.item()

            pbar.set_description("Current loss: {:.4f}".format(loss))
            pbar.update(1)

            if batch_idx == data_len: break
    
    # total, phone, pitch=None
    return total_loss / data_len, total_loss / data_len, None

def predict_w_bdr(args, ac_model, bdr_model, test_data, device, alpha, model_type):

    if not os.path.exists(args.pred_dir):
        os.makedirs(args.pred_dir)

    dataloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             collate_fn=utils.my_collate)
    ac_model.eval()
    bdr_model.eval()

    resolution = 256 / 22050 * 3

    with tqdm(total=len(test_data)) as pbar, torch.no_grad():
        for example_num, _data in enumerate(dataloader):
            x, idx, meta = _data
            line_start = [d[0] for d in idx[0][1]] # first sample in the batch (which has only one sample); second element in the tuple (align_idx, line_idx)
            idx = idx[0][0] # first sample in the batch (which has only one sample); first element in the tuple (align_idx, line_idx)
            words, audio_name, audio_length = meta[0]

            # reshape input, prepare mel
            x = x.reshape(1,1,-1)
            x = utils.move_data_to_device(x, device)
            x = x.squeeze(0)
            x = x.squeeze(1)
            x = train_audio_transforms.to(device)(x)
            x = nn.utils.rnn.pad_sequence(x, batch_first=True).unsqueeze(1)

            # predict
            ac_outputs = ac_model(x)
            if model_type == "MTL":
                ac_outputs = torch.sum(ac_outputs, dim=3)
            ac_outputs = F.log_softmax(ac_outputs, dim=2)

            # get boundary prob curve
            bdr_outputs = bdr_model(x).data.cpu().numpy().reshape(-1)
            # write boundary probabilities (after sigmoid)
            with open(os.path.join(args.pred_dir, audio_name + "_bdr.csv"), 'w') as f:
                for j in range(len(bdr_outputs)):
                    f.write("{},{}\n".format(j * resolution, bdr_outputs[j]))
            # apply log
            bdr_outputs = np.log(bdr_outputs) * alpha

            batch_num, output_length, num_classes = ac_outputs.shape
            song_pred = ac_outputs.data.cpu().numpy().reshape(-1, num_classes)
            total_length = int(audio_length / args.sr // resolution)
            song_pred = song_pred[:total_length, :] # total_length, num_classes

            # smoothing
            P_noise = np.random.uniform(low=1e-11, high=1e-10, size=song_pred.shape)
            song_pred = np.log(np.exp(song_pred) + P_noise)

            # dynamic programming alignment with boundary information
            word_align, score = utils.alignment_bdr(song_pred, words, idx, bdr_outputs, line_start)
            print("\t{}:\t{}".format(audio_name, score))

            # write
            with open(os.path.join(args.pred_dir, audio_name + "_align.csv"), 'w') as f:
                for j in range(len(word_align)):
                    word = word_align[j]
                    f.write("{},{},{}\n".format(word[0] * resolution, word[1] * resolution, words[idx[j,0]:idx[j,1]]))

            pbar.update(1)

    return 0