# DIV2K-loader
Python import functions and dataset class for DIV2K dataset.

See `explore_DIV2K.ipynb` for more details.

## Contributors

- Gastaldi Paolo (_paologastaldi-polito_)
- Gennero Stefano (_Stevezbiz_)

## Links

- DIV2K dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/

## List of subsets

For each subset you can require `train`, `valid` (validation) or `all` (train and validation).

- `bicubic_x2`
- `unknown_x2`
- `bicubic_x3`
- `unknown_x3`
- `bicubic_x4`
- `unknown_x4`
- `bicubic_x8`
- `mild_x4`
- `difficult_x4`
- `wild_x4`
- `HR_images`

## Folders structure

Downloaded data will be organized in this way:

```
/data
    /DIV2K
        /train
            /DIV2K_train_LR_bicubic
                /X2
                ...
            ...
        /valid
            /DIV2K_valid_LR_bicubic
                /X2
                ...
            ...
```

## Install dependencies

```
pip3 install torch
pip3 install wget
```

## Example

```
# Define interested subsets
subsets = {'bicubic_x2' : 'all', 'bicubic_x4' : 'train', 'unknown_x4' : 'valid'}

# Download data
DIV2KImport(sets=subsets)

# Create the dataset
dataset = DIV2KImageDataset(subsets=subsets)
```