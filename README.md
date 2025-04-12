# Dithionite-Assays

Dithionite assays were performed to model the membrane permeability of lipid membranes.
Lipid vesicles were prepared from various fatty acids and phospholipids with the fluorophore - NBD - tagged phospholipid DOPE at constant concentration 6 uM.
The lipid bilayer contains approximately half of the NBD inside of the bilayer and half outside since the tag is on the headgroup.
Triton was added to half the samples in order to disrupt the bilayer and micellize the vesicles exposing 100% of the NBD to the buffer solution.
Dithionite was added to the samples immediately before fluorescence spectroscopy was performed, the dithionite reduced NBD to the non-fluorescent ABD.
In the triton samples the rate of fluorescence decay should represent the reaction rate.
In the untreated samples the rate of fluorescence decay depends on both the reaction rate, and the rate of permeation of dithionite into the vesicles.

## Folder structure
In this repo, it has 3 subfolders: 'data', 'code', and 'results'.

### data
Data is stored in csv.\
Time is the first column.\
Natural log calculations and gwere included following the fluorescence data for manual graphing.\
CSV name includes concentration.
* `sample_25.csv`:  Needs the concentration

### code
This folder has 2 subfolders: 'src', 'notebooks'.
In 'src' folder, it include the main functions for modeling, plotting, and storing results
In 'notebooks' folder, it contains some example code of modeling data

Basic usage:
```
# Read data file
file_path2 = '../../data/33_mM_Decanoic.csv'

# Preprocess data
df_preprocess, filename = preprocess_data(file_path2)
data = df_preprocess

# Analysis and modeling
results21 = analyze_fluorescence_decay_triton(data, filename, time_range=60)
results21;
results22 = analyze_fluorescence_decay_no_triton(data, filename, p0 =[25, 25, 0.01, 0.001, 0.005])
results22;
results23 = analyze_fluorescence_decay_no_triton_numerical(data, filename)
results23;
```

### results
#### Manual results
Processed data is stored in excel workbooks.\
Natural log calculations were included following the fluorescence data for manual graphing.

#### Auto results
The modeling results are stored in separate folder, which the folder name indicate the dataset used and the creation time. e.g. 0.16_mM_DOPC_20250409-17:55

## Requirements for libraries
- numpy
- scipy
- pandas
- matplotlib
- scikit-learn


## Citation

If you use this repository in your research, please cite it as follows: 
TBD

## License

This project is licensed under the [BSD 3-Clause License](LICENSE). Please cite this repository if you use it in your work.
