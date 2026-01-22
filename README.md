# UITron-Speech

## 🎬 Introduction

This repository is the official implementation of UITron-Speech: Towards Automated GUI Agents Based on Speech Instructions.
![demo](asset/demo.gif)

<!-- <video src="https://github.com/GUIRoboTron/UITron-Speech/asset/demo.mp4"  width="60%" controls autoplay controls>
</video> -->



## **✨**News and ToDo List


- [ ] Release data
- [X] Release checkpoints
- [X] [2025-06-12] Release code



## 🎙️ Speech Dataset 
ChatTTS synthesizes speech for the Aguvis and OS-Atlas training sets, as well as the evaluation benchmarks (ScreenSpot, AndroidControl, GUI Odyssey). The example below illustrates the conversion of raw text instructions from Aguvis.

```bash
cd uitron-speech-data
```

``` python
python -m pipline.aguvis_tts.py
```

This script processes the original metadata to assign unique IDs to text prompts and exports an updated JSON file. It then utilizes ChatTTS to synthesize these prompts into speech, saving the audio files named with their corresponding IDs.

## 🎈 Checkpoint weights

UITron-Speech builds upon Qwen2.5-Omni and follows a two-stage training pipeline consisting of grounding and planning. The model weights from the stage2 are available for download at https://huggingface.co/hj611/qwen25_omni_stage2_d1.



## 👥 Citation
If you find our work valuable, we would appreciate your citation:
```text
@article{han2025guirobotron,
  title={GUIRoboTron-Speech: Towards Automated GUI Agents Based on Speech Instructions},
  author={Han, Wenkang and Zeng, Zhixiong and Huang, Jing and Jiang, Shu and Zheng, Liming and Yang, Longrong and Qiu, Haibo and Yao, Chang and Chen, Jingyuan and Ma, Lin},
  journal={arXiv preprint arXiv:2506.11127},
  year={2025}
}
```

## 🤗 Acknowledgements

Our codebase builds on [ms-swift](https://github.com/modelscope/ms-swift) and [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni). We also utilize [ChatTTS](https://github.com/2noise/ChatTTS) to generate speech instruction data for training.  Thanks the authors for sharing their awesome codebases!


