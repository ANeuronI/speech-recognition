to convert voice data from common voice to wav and create test and train file -->


python3 commonvoice_create_jsons.py --file_path "dataset/validated.tsv" --save_json_path "dataset/datafile/" --percent 20 --convert


for trainig -->

python3 neural\ network/train.py --train_file "/mnt/c/Users/yoshi/VPROJECTS/speech recog AIHT/dataset/datafile/train.json" --valid_file "/mnt/c/Users/yoshi/VPROJECTS/speech recog AIHT/dataset/datafile/test.json" --load_model_from "path to ckpt file"


to make frozan pytorch model after the training --->

python3 neural\ network/optimize_graph.py --model_checkpoint "/mnt/c/Users/yoshi/VPROJECTS/speech recog AIHT/tb_logs/speech_recognition/version_0/checkpoints/epoch=499-step=13987.ckpt" --save_path "/mnt/c/Users/yoshi/VPROJECTS/speech recog AIHT/freezemodel/model"


to start engine -->

python3 engine.py --model_file "/mnt/c/Users/yoshi/VPROJECTS/speech recog AIHT/neural network/freezemodel/model" 