# currently work in progress
## Setup and Usage Guide

This guide will walk you through the steps to convert voice data, train a model, optimize it, and start the engine for your speech recognition project.

### 1. Convert Voice Data

To convert voice data from Common Voice format to WAV and create test and train files, run:

```bash
python3 commonvoice_create_jsons.py \
  --file_path "dataset/validated.tsv" \
  --save_json_path "dataset/datafile/" \
  --percent 20 \
  --convert
```

### 2. Train the Model

To train your model, execute:

```bash
python3 neural_network/train.py \
  --train_file "/mnt/c/Users/yoshi/VPROJECTS/speech_recog_AIHT/dataset/datafile/train.json" \
  --valid_file "/mnt/c/Users/yoshi/VPROJECTS/speech_recog_AIHT/dataset/datafile/test.json" \
  --load_model_from "path_to_ckpt_file"
```

### 3. Optimize and Freeze the Model

After training, freeze your PyTorch model by running:

```bash
python3 neural_network/optimize_graph.py \
  --model_checkpoint "/mnt/c/Users/yoshi/VPROJECTS/speech_recog_AIHT/tb_logs/speech_recognition/version_0/checkpoints/epoch=499-step=13987.ckpt" \
  --save_path "/mnt/c/Users/yoshi/VPROJECTS/speech_recog_AIHT/freezemodel/model"
```

### 4. Start the Engine

Finally, to start the engine, use:

```bash
python3 engine.py \
  --model_file "/mnt/c/Users/yoshi/VPROJECTS/speech_recog_AIHT/neural_network/freezemodel/model"
```

---

Feel free to customize the paths and filenames according to your project's specifics. If you encounter any issues or have further questions, don't hesitate to reach out!

