# Multimodal Classification

## Setup

### Requirements

+ `torch`

+ `cv2`

+ `librosa`

+ `numpy`

+ `pydub`

```bash
conda install numpy
conda install -c pytorch pytorch
conda install -c conda-forge opencv librosa pydub
```

### Train Sequence

```bash
python video_gen.py -i videos -o frames
```

```bash
python audio_gen.py -i videos -o spectrograms
```

```bash
python model.py -f frames -s spectrograms
```

## More

Loophole: Transforms passed into DataLoaders (as argument) might not work