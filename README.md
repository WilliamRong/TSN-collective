# TSN-Pytorch

*Now in experimental release, suggestions welcome*.

**Note**: always use `git clone --recursive https://github.com/yjxiong/tsn-pytorch` to clone this project. 
Otherwise you will not be able to use the inception series CNN archs. 

This is a reimplementation of temporal segment networks (TSN) in PyTorch. All settings are kept identical to the original caffe implementation.

For optical flow extraction and video list generation, you still need to use the original [TSN codebase](https://github.com/yjxiong/temporal-segment-networks).

## Training

To train a new model, use the `main.py` script.

The command to reproduce the original TSN experiments of RGB modality on UCF101 can be 

```bash
python main.py ucf101 RGB <ucf101_rgb_train_list> <ucf101_rgb_val_list> \
   --arch BNInception --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
   -b 128 -j 8 --dropout 0.8 \
   --snapshot_pref ucf101_bninception_ 
```

For flow models:

```bash
python main.py ucf101 Flow <ucf101_flow_train_list> <ucf101_flow_val_list> \
   --arch BNInception --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 190 300 --epochs 340 \
   -b 128 -j 8 --dropout 0.7 \
   --snapshot_pref ucf101_bninception_ --flow_pref flow_  
```

For RGB-diff models:

```bash
python main.py ucf101 RGBDiff <ucf101_rgb_train_list> <ucf101_rgb_val_list> \
   --arch BNInception --num_segments 7 \
   --gd 40 --lr 0.001 --lr_steps 80 160 --epochs 180 \
   -b 128 -j 8 --dropout 0.8 \
   --snapshot_pref ucf101_bninception_ 
```

## Testing

After training, there will checkpoints saved by pytorch, for example `ucf101_bninception_rgb_checkpoint.pth`.

Use the following command to test its performance in the standard TSN testing protocol:

```bash
python test_models.py ucf101 RGB <ucf101_rgb_val_list> ucf101_bninception_rgb_checkpoint.pth \
   --arch BNInception --save_scores <score_file_name>

```

Or for flow models:
 
```bash
python test_models.py ucf101 Flow <ucf101_rgb_val_list> ucf101_bninception_flow_checkpoint.pth \
   --arch BNInception --save_scores <score_file_name> --flow_pref flow_

```

##Score
One can also use cached score files to evaluate the performance. To do this, issue the following command

```
python tools/eval_scores.py SCORE_FILE
```

The more important function of `eval_scores.py` is to do modality fusion.
For example, once we got the scores of rgb stream in `RGB_SCORE_FILE` and flow stream in `FLOW_SCORE_FILE`.
The fusion result with weights of `1:1.5` can be achieved with

```
python tools/eval_scores.py RGB_SCORE_FILE FLOW_SCORE_FILE --score_weights 1 1.5
```

To view the full help message of these scripts, run `python eval_net.py -h` or `python eval_scores.py -h`. 
