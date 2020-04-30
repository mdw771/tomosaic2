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

## Example dataset
We have made a working example dataset (the charcoal data shown in the paper) public on TomoBank, so that anyone can grab the data for a trial run! Data can be downloaded from [here](https://tomobank.readthedocs.io/en/latest/source/data/docs.data.tomosaic.html).

## How to use
The most convenient way to use Tomosaic is through [Automo](https://github.com/decarlof/automo), a higher-level wrapper for Tomosaic which also includes a collection of script files to be used in the Tomosaic workflow. If you would rather not use Automo, move on to the following:

Get the tomosaic script templates from [here](https://github.com/mdw771/tomosaic_templates). Again, you can do
```
git clone https://github.com/mdw771/tomosaic_templates
```
Copy the Python scripts to the same directory where you save all the HDF5 data files. Then, follow these steps for mosaic reconstruction:

0. Make sure all HDF5 files are named in the pattern of `prefix_y?_x?.h5`. The indices of tiles should start from 0, from left to right, top to bottom of the collection positions on the sample. 

1. Run `python mosaic_reorganize` in the data directory. This will put all original HDF5 files in a folder called `data_raw_1x`; if you want to have downsampled copies of the data, use the `--ds` flag to indicate the downsampling ratios (remember to include 1), e.g., `--ds 1,2,4`.

2. Open `mosaic_meta.py`, and enter the filename prefix, and estimated y and x offset (in pixel). Change `data_format` if necessary; if you are unsure of how to refer to your data format, see dxchange`https://github.com/data-exchange/dxchange`'s documentation. Use `python mosaic_meta.py` to run this script. 

3. Run `python mosaic_preview.py --pano 0; python mosaic_preview.py --pano <last_frame>` (If a total of 2000 projections are collected, then `<last_frame>` should be 1999). This creates the stitched panorama at 0 and 180 degrees, saved in the `preview_panos` folder. 

4. Run `python mosaic_find_center.py`. It will write reconstructions generated with a series of rotation center settings (+/- 5 from the phase correlation guess, calculated using the 0 and 180 projections generated in step 3) into `./center`. You can then check those output files to manually refine the center. 

5. Open `center_pos.txt` which is created automatically in the last step. If a phase correlation guess was done, the file should contain something like
```
0 xxxx
1 xxxx
```
where the first integer at each line is the grid row index, and the second number is the center value. Change the values to your manually refined results (if applicable). 

6. Open `mosaic_recon.py`, set `slice_st` and `slice_end`, make sure `mode` is `discrete`, and run `python mosaic_recon.py`.

(Note: Mode `discrete` stitches sinograms for each slice instead of doing x-y stitching for each projection angle. The latter is called the `merged` mode and requires you to create a merged HDF5 file before reconstruction. This can be done using `mosaic_merge.py`. For details, please read the Automo documentation.)

## Publications
Vescovi, R. et al. Tomosaic: efficient acquisition and reconstruction of teravoxel tomography data using limited-size synchrotron X-ray beams. *Journal of Synchrotron Radiation* **25**, (2018).
  
