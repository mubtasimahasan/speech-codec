from transformers import HubertModel, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Tokenizer, AutoModel, AutoTokenizer
from pathlib import Path
import torchaudio
import torch
from tqdm import tqdm
import random
import numpy as np
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, help='Audio folder path')
    parser.add_argument('--rep_typ', type=str, help='Representation type: "hubert", "llm", or "combined"')
    parser.add_argument('--exts', type=str, help="Audio file extensions, splitting with ','", default='flac')
    parser.add_argument('--split_seed', type=int, help="Random seed", default=42)
    parser.add_argument('--valid_set_size', type=float, default=1000)
    args = parser.parse_args()

    exts = args.exts.split(',')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Configuration settings
    sample_rate = 16000
    semantic_model_path = "facebook/hubert-base-ls960"
    semantic_model_layer = "avg"
    stt_model_path = "facebook/wav2vec2-base-960h"
    llm_model_path = "bert-base-uncased"
    segment_size = 48000
    train_file_list = f"{args.rep_typ}_train_file_list.txt"
    valid_file_list = f"{args.rep_typ}_dev_file_list.txt"


    # Hubert Model and Feature Extractor
    if args.rep_typ in ['hubert', 'combined']:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(semantic_model_path)
        hubert_model = HubertModel.from_pretrained(semantic_model_path).eval().to(device)

    # LLM Model, Tokenizer, and STT Model
    if args.rep_typ in ['llm', 'combined']:
        stt_model = Wav2Vec2ForCTC.from_pretrained(stt_model_path).eval().to(device)
        stt_tokenizer = Wav2Vec2Tokenizer.from_pretrained(stt_model_path)
        llm_model = AutoModel.from_pretrained(llm_model_path).eval().to(device)
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)

    path = Path(args.audio_dir)
    file_list = [str(file) for ext in exts for file in path.glob(f'**/*.{ext}')]

    if args.valid_set_size != 0 and args.valid_set_size < 1:
        valid_set_size = int(len(file_list) * args.valid_set_size)
    else:
        valid_set_size = int(args.valid_set_size)

    random.seed(args.split_seed)
    random.shuffle(file_list)
    print(f'A total of {len(file_list)} samples will be processed, and {valid_set_size} of them will be included in the validation set.')

    with torch.no_grad():
        for i, audio_file in tqdm(enumerate(file_list)):
            wav, sr = torchaudio.load(audio_file)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            if wav.size(-1) < segment_size:
                print("@@@@@@@@ wav.size(-1) < segment_size!! Check if dataset was processed correctly.")
                wav = torch.nn.functional.pad(wav, (0, segment_size - wav.size(-1)), 'constant')

            # Extract Hubert representation if needed
            if args.rep_typ in ['hubert', 'combined']:
                input_values = feature_extractor(wav.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_values
                hubert_output = hubert_model(input_values.to(hubert_model.device), output_hidden_states=True)
                if semantic_model_layer == 'avg':
                    hubert_rep = torch.mean(torch.stack(hubert_output.hidden_states), axis=0)
                else:
                    hubert_rep = hubert_output.hidden_states[semantic_model_layer]
                hubert_rep_file = audio_file.replace(args.audio_dir, 'hubert_rep').split('.')[0] + '.hubert.npy'
                hubert_rep_sub_dir = '/'.join(hubert_rep_file.split('/')[:-1])
                if not os.path.exists(hubert_rep_sub_dir):
                    os.makedirs(hubert_rep_sub_dir)
                np.save(hubert_rep_file, hubert_rep.detach().cpu().numpy())

            # Extract LLM representation if needed
            if args.rep_typ in ['llm', 'combined']:
                # Step 1: Convert audio to text
                input_values = stt_tokenizer(wav.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_values
                logits = stt_model(input_values.to(stt_model.device)).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = stt_tokenizer.batch_decode(predicted_ids)[0]
                # Step 2: Pass text through LLM
                llm_inputs = llm_tokenizer(transcription, return_tensors="pt", truncation=True,
                                           padding="max_length", max_length=logits.shape[1]).to(device)
                llm_outputs = llm_model(**llm_inputs, output_hidden_states=True)
                # Step 3: Extract representation from target layer
                if semantic_model_layer == 'avg':
                    llm_rep = torch.mean(torch.stack(llm_outputs.hidden_states), axis=0)
                else:
                    llm_rep = llm_outputs.hidden_states[semantic_model_layer]
                # Step 4: Save the representation
                llm_rep_file = audio_file.replace(args.audio_dir, 'llm_rep').split('.')[0] + '.llm.npy'
                llm_rep_sub_dir = '/'.join(llm_rep_file.split('/')[:-1])
                if not os.path.exists(llm_rep_sub_dir):
                    os.makedirs(llm_rep_sub_dir)
                np.save(llm_rep_file, llm_rep.detach().cpu().numpy())

            # Write to file list based on chosen representation
            if args.rep_typ == 'hubert':
                rep_line = audio_file + "\t" + hubert_rep_file + "\n"
            elif args.rep_typ == 'llm':
                rep_line = audio_file + "\t" + llm_rep_file + "\n"
            elif args.rep_typ == 'combined':
                rep_line = audio_file + "\t" + hubert_rep_file + "\t" + llm_rep_file + "\n"

            if i == 0 or i == valid_set_size:
                with open(valid_file_list if i < valid_set_size else train_file_list, 'w') as f:
                    f.write(rep_line)
            else:
                with open(valid_file_list if i < valid_set_size else train_file_list, 'a+') as f:
                    f.write(rep_line)
