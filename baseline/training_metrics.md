## Undistorted

### k_0

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/undistorted/k_0" \
          -p="data/train/undistorted/k_0" \
          -t="data/train/undistorted/k_0" \
          -num_iterations=0 \
          -o="out/undistorted/k_0"
```

**OUTPUT**

```txt
optimal p1/p2: 0.43 0.45
optimal score: {'auc': np.float64(0.588), 'c@1': 0.535, 'f_05_u': 0.557, 'F1': np.float64(0.647), 'brier': np.float64(0.752), 'overall': np.float64(0.616)}
-> determining optimal threshold
Dev results -> F1=0.6651324289405685 at th=0.25
```

### k_1

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/undistorted/k_1" \
          -p="data/train/undistorted/k_1" \
          -t="data/train/undistorted/k_1" \
          -num_iterations=0 \
          -o="out/undistorted/k_1"
```

**OUTPUT**

```txt
optimal p1/p2: 0.01 0.43
optimal score: {'auc': np.float64(0.584), 'c@1': 0.498, 'f_05_u': 0.543, 'F1': np.float64(0.681), 'brier': np.float64(0.751), 'overall': np.float64(0.612)}
-> determining optimal threshold
Dev results -> F1=0.6677383057386272 at th=0.25
```

### k_2

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/undistorted/k_2" \
          -p="data/train/undistorted/k_2" \
          -t="data/train/undistorted/k_2" \
          -num_iterations=0 \
          -o="out/undistorted/k_2"
```

**OUTPUT**

```txt
optimal p1/p2: 0.43 0.45
optimal score: {'auc': np.float64(0.582), 'c@1': 0.526, 'f_05_u': 0.548, 'F1': np.float64(0.637), 'brier': np.float64(0.751), 'overall': np.float64(0.609)}
-> determining optimal threshold
Dev results -> F1=0.6596573841032719 at th=0.263013013013013
```

### k_3

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/undistorted/k_3" \
          -p="data/train/undistorted/k_3" \
          -t="data/train/undistorted/k_3" \
          -num_iterations=0 \
          -o="out/undistorted/k_3"
```

**OUTPUT**

```txt
optimal p1/p2: 0.11 0.43
optimal score: {'auc': np.float64(0.584), 'c@1': 0.493, 'f_05_u': 0.538, 'F1': np.float64(0.676), 'brier': np.float64(0.751), 'overall': np.float64(0.609)}
-> determining optimal threshold
Dev results -> F1=0.6650016113438608 at th=0.4857357357357357
```

### k_4

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/undistorted/k_4" \
          -p="data/train/undistorted/k_4" \
          -t="data/train/undistorted/k_4" \
          -num_iterations=0 \
          -o="out/undistorted/k_4"
```

**OUTPUT**

```txt
optimal p1/p2: 0.42 0.44
optimal score: {'auc': np.float64(0.586), 'c@1': 0.525, 'f_05_u': 0.554, 'F1': np.float64(0.647), 'brier': np.float64(0.751), 'overall': np.float64(0.613)}
-> determining optimal threshold
Dev results -> F1=0.6648356086921399 at th=0.2535035035035035 
```

## DV-MA-k-300

### k_0

**INPUT**

```bash
python3 cngdist.py \
          --train \
          --model_dir="models/baseline/DV-MA-k-300/k_0" \
          -p="data/train/DV-MA-k-300/k_0" \
          -t="data/train/DV-MA-k-300/k_0" \
          -num_iterations=0 \
          -o="out/DV-MA-k-300/k_0" > DV-MA-k-300_k_0_CLIOUT.txt && \
```

**OUTPUT**

```txt
optimal p1/p2: 0.01 0.98
optimal score: {'auc': np.float64(0.528), 'c@1': 0.486, 'f_05_u': 0.532, 'F1': np.float64(0.672), 'brier': np.float64(0.726), 'overall': np.float64(0.589)}
-> determining optimal threshold
Dev results -> F1=0.668005946402025 at th=0.25
```

### k_1

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/DV-MA-k-300/k_1" \
          -p="data/train/DV-MA-k-300/k_1" \
          -t="data/train/DV-MA-k-300/k_1" \
          -num_iterations=0 \
          -o="out/DV-MA-k-300/k_1"
```

**OUTPUT**

```txt
optimal p1/p2: 0.01 0.98
optimal score: {'auc': np.float64(0.531), 'c@1': 0.488, 'f_05_u': 0.534, 'F1': np.float64(0.674), 'brier': np.float64(0.727), 'overall': np.float64(0.591)}
-> determining optimal threshold
Dev results -> F1=0.6674169949352842 at th=0.25
```

### k_2

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/DV-MA-k-300/k_2" \
          -p="data/train/DV-MA-k-300/k_2" \
          -t="data/train/DV-MA-k-300/k_2" \
          -num_iterations=0 \
          -o="out/DV-MA-k-300/k_2"
```

**OUTPUT**

```txt
optimal p1/p2: 0.01 0.98
optimal score: {'auc': np.float64(0.532), 'c@1': 0.486, 'f_05_u': 0.532, 'F1': np.float64(0.672), 'brier': np.float64(0.727), 'overall': np.float64(0.59)}
-> determining optimal threshold
Dev results -> F1=0.6670955291090382 at th=0.25
```

### k_3

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/DV-MA-k-300/k_3" \
          -p="data/train/DV-MA-k-300/k_3" \
          -t="data/train/DV-MA-k-300/k_3" \
          -num_iterations=0 \
          -o="out/DV-MA-k-300/k_3"
```

**OUTPUT**

```txt
optimal p1/p2: 0.01 0.98
optimal score: {'auc': np.float64(0.524), 'c@1': 0.481, 'f_05_u': 0.527, 'F1': np.float64(0.668), 'brier': np.float64(0.725), 'overall': np.float64(0.585)}
-> determining optimal threshold
Dev results -> F1=0.6652709557935421 at th=0.25
```

### k_4

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/DV-MA-k-300/k_4" \
          -p="data/train/DV-MA-k-300/k_4" \
          -t="data/train/DV-MA-k-300/k_4" \
          -num_iterations=0 \
          -o="out/DV-MA-k-300/k_4"
```

**OUTPUT**

```txt
optimal p1/p2: 0.01 0.97
optimal score: {'auc': np.float64(0.526), 'c@1': 0.499, 'f_05_u': 0.551, 'F1': np.float64(0.667), 'brier': np.float64(0.685), 'overall': np.float64(0.586)}
-> determining optimal threshold
Dev results -> F1=0.6655395886165117 at th=0.25
```

## DV-MA-k-3000

### k_0

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/DV-MA-k-3000/k_0" \
          -p="data/train/DV-MA-k-3000/k_0" \
          -t="data/train/DV-MA-k-3000/k_0" \
          -num_iterations=0 \
          -o="out/DV-MA-k-3000/k_0"
```

**OUTPUT**

```txt
optimal p1/p2: 0.88 0.91
optimal score: {'auc': np.float64(0.548), 'c@1': 0.512, 'f_05_u': 0.548, 'F1': np.float64(0.66), 'brier': np.float64(0.714), 'overall': np.float64(0.597)}
-> determining optimal threshold
Dev results -> F1=0.6675793203414399 at th=0.4271771771771772
```

### k_1

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/DV-MA-k-3000/k_1" \
          -p="data/train/DV-MA-k-3000/k_1" \
          -t="data/train/DV-MA-k-3000/k_1" \
          -num_iterations=0 \
          -o="out/DV-MA-k-3000/k_1"
```

**OUTPUT**

```txt
optimal p1/p2: 0.89 0.91
optimal score: {'auc': np.float64(0.55), 'c@1': 0.517, 'f_05_u': 0.551, 'F1': np.float64(0.655), 'brier': np.float64(0.715), 'overall': np.float64(0.598)}
-> determining optimal threshold
Dev results -> F1=0.6677676629340208 at th=0.4271771771771772
```

### k_2

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/DV-MA-k-3000/k_2" \
          -p="data/train/DV-MA-k-3000/k_2" \
          -t="data/train/DV-MA-k-3000/k_2" \
          -num_iterations=0 \
          -o="out/DV-MA-k-3000/k_2"
```

**OUTPUT**

```txt
optimal p1/p2: 0.01 0.9
optimal score: {'auc': np.float64(0.543), 'c@1': 0.499, 'f_05_u': 0.544, 'F1': np.float64(0.674), 'brier': np.float64(0.703), 'overall': np.float64(0.593)}
-> determining optimal threshold
Dev results -> F1=0.6659691762906925 at th=0.25
```

### k_3

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/DV-MA-k-3000/k_3" \
          -p="data/train/DV-MA-k-3000/k_3" \
          -t="data/train/DV-MA-k-3000/k_3" \
          -num_iterations=0 \
          -o="out/DV-MA-k-3000/k_3"
```

**OUTPUT**

```txt
optimal p1/p2: 0.87 0.91
optimal score: {'auc': np.float64(0.55), 'c@1': 0.51, 'f_05_u': 0.547, 'F1': np.float64(0.667), 'brier': np.float64(0.715), 'overall': np.float64(0.598)}
-> determining optimal threshold
Dev results -> F1=0.6668817811479046 at th=0.4436936936936937
```

### k_4

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/DV-MA-k-3000/k_4" \
          -p="data/train/DV-MA-k-3000/k_4" \
          -t="data/train/DV-MA-k-3000/k_4" \
          -num_iterations=0 \
          -o="out/DV-MA-k-3000/k_4"
```

**OUTPUT**

```txt
optimal p1/p2: 0.78 0.91
optimal score: {'auc': np.float64(0.55), 'c@1': 0.499, 'f_05_u': 0.543, 'F1': np.float64(0.68), 'brier': np.float64(0.715), 'overall': np.float64(0.598)}
-> determining optimal threshold
Dev results -> F1=0.6684918270392141 at th=0.49024024024024027
```

## DV-MA-k-20000

### k_0

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/DV-MA-k-20000/k_0" \
          -p="data/train/DV-MA-k-20000/k_0" \
          -t="data/train/DV-MA-k-20000/k_0" \
          -num_iterations=0 \
          -o="out/DV-MA-k-20000/k_0"
```

**OUTPUT**

```txt
optimal p1/p2: 0.01 0.41000000000000003
optimal score: {'auc': np.float64(0.536), 'c@1': 0.501, 'f_05_u': 0.548, 'F1': np.float64(0.674), 'brier': np.float64(0.731), 'overall': np.float64(0.598)}
-> determining optimal threshold
Dev results -> F1=0.6682734795533061 at th=0.25
```

### k_1

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/DV-MA-k-20000/k_1" \
          -p="data/train/DV-MA-k-20000/k_1" \
          -t="data/train/DV-MA-k-20000/k_1" \
          -num_iterations=0 \
          -o="out/DV-MA-k-20000/k_1"
```

**OUTPUT**

```txt
optimal p1/p2: 0.01 0.41000000000000003
optimal score: {'auc': np.float64(0.537), 'c@1': 0.501, 'f_05_u': 0.547, 'F1': np.float64(0.673), 'brier': np.float64(0.731), 'overall': np.float64(0.598)}
-> determining optimal threshold
Dev results -> F1=0.6677383057386272 at th=0.25
```

### k_2

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/DV-MA-k-20000/k_2" \
          -p="data/train/DV-MA-k-20000/k_2" \
          -t="data/train/DV-MA-k-20000/k_2" \
          -num_iterations=0 \
          -o="out/DV-MA-k-20000/k_2"
```

**OUTPUT**

```txt
optimal p1/p2: 0.01 0.44
optimal score: {'auc': np.float64(0.539), 'c@1': 0.495, 'f_05_u': 0.54, 'F1': np.float64(0.675), 'brier': np.float64(0.734), 'overall': np.float64(0.597)}
-> determining optimal threshold
Dev results -> F1=0.6652172162499497 at th=0.25
```

### k_3

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/DV-MA-k-20000/k_3" \
          -p="data/train/DV-MA-k-20000/k_3" \
          -t="data/train/DV-MA-k-20000/k_3" \
          -num_iterations=0 \
          -o="out/DV-MA-k-20000/k_3"
```

**OUTPUT**

```txt
optimal p1/p2: 0.01 0.44
optimal score: {'auc': np.float64(0.537), 'c@1': 0.495, 'f_05_u': 0.541, 'F1': np.float64(0.676), 'brier': np.float64(0.734), 'overall': np.float64(0.597)}
-> determining optimal threshold
Dev results -> F1=0.6657007164131047 at th=0.25
```

### k_4

**INPUT**

```bash
$ python3 cngdist.py \
          --train \
          --model_dir="models/baseline/DV-MA-k-20000/k_4" \
          -p="data/train/DV-MA-k-20000/k_4" \
          -t="data/train/DV-MA-k-20000/k_4" \
          -num_iterations=0 \
          -o="out/DV-MA-k-20000/k_4"
```

**OUTPUT**

```txt
optimal p1/p2: 0.01 0.44
optimal score: {'auc': np.float64(0.542), 'c@1': 0.497, 'f_05_u': 0.543, 'F1': np.float64(0.678), 'brier': np.float64(0.735), 'overall': np.float64(0.599)}
-> determining optimal threshold
Dev results -> F1=0.666398487469327 at th=0.25
```