# AI-KU-HAR

Repository used to develop an LSTM-based solution for HAR.
Based on tensorflow (should be agnostic to version).

---
## Instructions

### Download and unzip the data:

```
 wget https://data.mendeley.com/public-files/datasets/45f952y38r/files/49c6120b-59fd-466c-97da-35d53a4be595/file_downloaded -O data.zip
 unzip data.zip -d data/
```

### Create CSV files:

```
python3 generate_dataset.py
```

### Train model:

```
python3 train.py
```

### Test model:

```
python3 test.py
```

### Convert model to TFLite for timing tests:

```
python3 convert.py
```

### Test for timing constraints:

```
python3 time-test.py
```
---
## Info

If you use this code, please cite us!

Silva, Mateus, et al. "Toward the design of a novel wearable system for field research in ecology." 2022 XII Brazilian Symposium on Computing Systems Engineering (SBESC). IEEE, 2022.
