# Tomosaic

Tomosaic is a library for Panoramic or Mosaic Tomography.\

## Installation
Get this repository to your hard drive using 
```
git clone https://github.com/mdw771/tomosaic2
```
and then use PIP to build and install:
```
pip install -e ./tomosaic2
```

## How to use
Get the tomosaic script templates from here(https://github.com/mdw771/tomosaic_templates). Again, you can do
```
git clone https://github.com/mdw771/tomosaic_templates
```
Copy the Python scripts to the same directory where you save all the HDF5 data files. Then, follow these steps for mosaic reconstruction:

0. Make sure all HDF5 files are named in the pattern of `prefix_y?_x?.h5`. The indices of tiles should start from 0, from left to right, top to bottom of the collection positions on the sample. 

1. Run `python mosaic_reorganize` in the data directory. This will put all original HDF5 files in a folder called `data_raw_1x`; if you want to have downsampled copies of the data, use the `--ds` flag to indicate the downsampling ratios (remember to include 1), e.g., `--ds 1,2,4`.

1. Open `mosaic_meta.py`, and enter the filename prefix, and estimated y and x offset (in pixel). Change `data_format` if necessary; if you are unsure of how to refer to your data format, see dxchange`https://github.com/data-exchange/dxchange`'s documentation. Use `python mosaic_meta.py` to run this script. 

2. 
