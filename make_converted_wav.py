import os
from inference_copy import inference
from itertools import permutations
from tqdm import tqdm
import torch

#converted_all
#unseen_speakers = ['p252', 'p261', 'p241', 'p238', 'p243', 'p294', 'p334', 'p343', 'p360', 'p362']

###unseen_sentences = ['002', '003', '004', '005', '006', '007', '009', '010', '011', '012']
unseen_sentences = ['002', '003', '004', '005']



#converted_test
#seen_speakers = ['p229','p238','p243','p252']
#unseen_speakers = ['p261','p360','p294','p241']
#unseen_sentences = ['001', '002', '003', '004', '005']


#seen_speakers = ['p229','p238','p243','p252']
#seen_speakers = ['p229','p243']
seen_speakers = ['p236','p259', 'p239','p263']

model_name = 'last_train_dec_4speakers_original'

# real_last_cycle_train_dec_4speakers_all_cycle6_from_50
# real_last_cycle_train_dec_4speakers_cycle6_from_50
# real_last_cycle_train_dec_4speakers_original
# real_last_cycle_train_dec_4speakers_original_one_hot

# real_last_cycle_train_dec_4speakers_cycle6_from_0
# real_last_cycle_train_dec_4speakers_all_cycle6_from_0

# real_last_cycle_train_dec_4speakers_cycle6_from_100
# real_last_cycle_train_dec_4speakers_all_cycle6_from_100

# real_last_cycle_train_dec_4speakers_iii3_cycle6_from_50

for epoch in range(100, 301, 10):
    epoch = str(epoch)

    seenORunseen = 'seen'

    speakers = seen_speakers if seenORunseen == 'seen' else unseen_speakers
    output_last_dir = model_name + '_' + epoch

    test_path = '/home/rtrt505/speechst1/CycleDiffusion/VCTK_2F2M/wavs'
    vc_model_path = f'/home/rtrt505/speechst1/CycleDiffusion/{model_name}/vc_{epoch}_0810.pt'
    output_dir = f'/home/rtrt505/speechst1/CycleDiffusion/converted_all/{model_name}/'
    output_dir = os.path.join(output_dir, output_last_dir)

    for src_speaker, tgt_speaker in permutations(speakers, 2):
        for src_sentence, tgt_sentence in permutations(unseen_sentences, 2):
            src_path = os.path.join(test_path, src_speaker, f'{src_speaker}_{src_sentence}_mic1.wav')
            tgt_path = os.path.join(test_path, tgt_speaker, f'{tgt_speaker}_{tgt_sentence}_mic1.wav')
            
            output_path = os.path.join(output_dir, f'{src_speaker}_to_{tgt_speaker}', f'{src_sentence}_to_{tgt_sentence}.wav')
            if not os.path.exists(output_path):
                # 폴더가 없을 경우 생성
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))        

                # 파일이 존재하는지 확인 후 inference 호출
                if os.path.exists(src_path) and os.path.exists(tgt_path):
                    #torch.cuda.empty_cache()
                    inference(vc_model_path, src_path, tgt_path, output_path)
                    print(f"{src_speaker}_to_{tgt_speaker} and {src_sentence}_to_{tgt_sentence} complete")
                else:
                    continue
            else:
                continue