from typing import Optional, List
import argparse
import os
import sys

import json
import uuid
import concurrent.futures
from tqdm import tqdm

import numpy as np
import torch 
torch.set_float32_matmul_precision('high')

import ChatTTS
from tools.logger import get_logger
from tools.audio import pcm_arr_to_mp3_view, pcm_arr_to_wav_view
from tools.normalizer.en import normalizer_en_nemo_text
from tools.normalizer.zh import normalizer_zh_tn

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

logger = get_logger("Command")


def save_mp3_file(wav, index):
    data = pcm_arr_to_mp3_view(wav)
    mp3_filename = f"output_audio_{index}.mp3"
    with open(mp3_filename, "wb") as f:
        f.write(data)
    logger.info(f"Audio saved to {mp3_filename}")


def load_normalizer(chat: ChatTTS.Chat):
    # try to load normalizer
    try:
        chat.normalizer.register("en", normalizer_en_nemo_text())
    except ValueError as e:
        logger.error(e)
    except BaseException:
        logger.warning("Package nemo_text_processing not found!")
        logger.warning(
            "Run: conda install -c conda-forge pynini=2.1.5 && pip install nemo_text_processing",
        )
    try:
        chat.normalizer.register("zh", normalizer_zh_tn())
    except ValueError as e:
        logger.error(e)
    except BaseException:
        logger.warning("Package WeTextProcessing not found!")
        logger.warning(
            "Run: conda install -c conda-forge pynini=2.1.5 && pip install WeTextProcessing",
        )


def main(
    # texts: List[str],
    origin_path:str,
    output_path:str,
    bs:int,
    speakers:int,
    spk: Optional[str] = None,
    stream: bool = False,
    source: str = "local",
    custom_path: str = "",
):
    # logger.info("Text input: %s", str(texts))
    logger.info("Text input: %s", str(origin_path))
    chat = ChatTTS.Chat(get_logger("ChatTTS"))
    logger.info("Initializing ChatTTS...")
    load_normalizer(chat)

    is_load = False
    if os.path.isdir(custom_path) and source == "custom":
        is_load = chat.load(compile=True, source="custom", custom_path=custom_path)
    else:
        is_load = chat.load(compile=True, source=source)

    if is_load:
        logger.info("Models loaded successfully.")
    else:
        logger.error("Models load failed.")
        sys.exit(1)
    
    with open(origin_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    texts=[]
    uuid_ids=[]
    # Each 'case' represents a GUI interaction session
    # 'conversations' is a list
    for case in metadata:
        for item in case['conversations']:
            if item['from']=="human":
                text_prompt=item['value'].replace("<image>\n", "").strip()
                text_prompt=text_prompt+'[uv_break]'
                uuid4 = uuid.uuid4()
                item['speech_id']=str(uuid4)
                uuid_ids.append(item['speech_id'])
                texts.append(text_prompt)

    print(len(texts))

    with open(origin_path[:-5]+'_re'+'.json', "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    params_refine_text = ChatTTS.Chat.RefineTextParams(prompt='[oral_1][laugh_0][break_0]',)
    logger.info("Start inference.")
    os.makedirs(os.path.dirname(output_path),exist_ok=True)

    for i in tqdm(range(0, len(texts), bs),desc="processing text prompt:"): 
        text_batch = texts[i : i + bs]  
        uid_batch=uuid_ids[i : i + bs]
        spk = chat.sample_random_speaker()
        wavs = chat.infer( 
            text_batch,
            stream,
            params_infer_code=ChatTTS.Chat.InferCodeParams(
                spk_emb=spk,
                prompt="[speed_2]"
            ),
            params_refine_text=params_refine_text
        )

        wavs_list=[]
        for wav in wavs:
            wavs_list.append(wav)
                
        for wav,uid in zip(wavs_list,uid_batch):
            save_path = os.path.join(output_path, f"{uid}.mp3")
            data = pcm_arr_to_mp3_view(wav)
            with open(save_path, "wb") as f:
                f.write(data)


if __name__ == "__main__":
    r"""
    python -m examples.cmd.run \
        --source custom --custom_path ../../models/2Noise/ChatTTS "Hello World" ":)"
    """
    logger.info("Starting ChatTTS commandline demo...")
    parser = argparse.ArgumentParser(
        description="ChatTTS Command",
        usage='[--spk xxx] [--stream] [--source ***] [--custom_path XXX] "Your text 1." " Your text 2."',
    )
    parser.add_argument(
        "--spk",
        help="Speaker (empty to sample a random one)",
        type=Optional[str],
        default=None,
    )
    parser.add_argument(
        "--stream",
        help="Use stream mode",
        action="store_true",
    )
    parser.add_argument(
        "--source",
        help="source form [ huggingface(hf download), local(ckpt save to asset dir), custom(define) ]",
        type=str,
        default="custom",
    )
    parser.add_argument(
        "--custom_path",
        help="custom defined model path(include asset ckpt dir)",
        type=str,
        default="",
    )
    parser.add_argument(
        "--origin_path",
        help="Original text prompt, e.g. /xxxx/seeclick.json",
        default="",
    )
    parser.add_argument(
        "--bs",
        help="batch size",
        type=int,
        default=64,
    )    
    parser.add_argument(
        "--output_path",
        help="Output directory for speech files",
        default='',
    )    
    parser.add_argument(
        "--speakers",
        help="Number of speakers",
        type=int,
        default=10000,
    )    
    
    args = parser.parse_args()
    logger.info(args)
    main(args.origin_path,args.output_path,args.bs,args.speakers, args.spk, args.stream, args.source, args.custom_path)
    logger.info("ChatTTS process finished.")