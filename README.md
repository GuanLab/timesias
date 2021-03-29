# Timesias
Forcast outcomes from time-series history. This is the top-performing algorithm for [DII National Data Science Challenge](https://sbmi.uth.edu/news/story.htm?id=4a7fba5d-2bd9-402a-a3bb-a2f5d21d2fe3).

## Installation
Install this package via pip:
``` r
pip install timesias
```
or clone this program to your local directory: 

``` r
git clone https://github.com/GuanLab/timesias.git
```
## Dependency

* [python (3.7.4)](https://www.python.org/)
* [numpy (1.17.2)](https://numpy.org/)
* [pandas (0.25.1)](https://pandas.pydata.org/)
* [LightGBM (3.1.1)](https://pypi.org/project/lightgbm/)
* [scikit-learn (0.19.0)](https://scikit-learn.org/stable/) 
* [shap (0.38.1)](https://pypi.org/project/shap/)

For visualization:
* [boken (2.3.0)](https://docs.bokeh.org/en/latest/docs/first_steps/installation.html)

## Input data format

The example data in the data/ are randomly generated data for the demonstration of the algorithm.

Two types of data is requied for model training and prediction:
* `gs.file`: `.txt` file two columns. The first column is paths for time-series records. The second column is the gold standard (0/1), representing the final outbreak of sepsis

``` 
./data/0.psv,1
./data/1.psv,1
./data/2.psv,0
./data/3.psv,1
./data/4.psv,0
./data/5.psv,1
./data/6.psv,1
``` 

* `*.psv`: `.psv` table files separated by `|`, which are the time-series records.
	The header of psv file are the feature names. To note, the first column is the time index.

``` 
HR|feature_1|feature_2|...|feature_n-1|feature_n
0.0|1|0.0|...|1.3|0.0 
1.0|NaN|0.0|...|0.0|0.0
3.5|NaN|2.3|...|0.0|0.0
```
## Model training and cross validation
``` r
timesias -g [GS_FILE_PATH] -t [LAST_N_RECORDS] -f [EXTRA_FEATURES] --shap
```

* `GS_FILE_PATH`: the path to the gold-standard file; for example, `/data/gs.file`;
* `LAST_N_RECORDS`: last n records to use for prediction. default: 16;
* `EXTRA_FEATURES`: addtional features used for prediction. default: ['norm', 'std', 'missing_portion', 'baseline'], which are all features we used in DII Data challenge.

also use

```r
 timesias --help
```
to get instructions on the usage of our program.


This will generate models, which will be saved under a new directory `./models`.

This will also create a new folder `./results`, where results mentioned below will be stored.

Evaluation results during five-fold cross validation will be stored in `./results/eva.tsv`.


# Top feature evaluation

if `--shap` is indicated, SHAP analysis will be carried out to show top contributing measurements and last nth time points. This will generate an html report (`./results/top_feature_report.html`) like the following:

<p align="center">
<img width="800", src ="https://github.com/GuanLab/timesias/blob/master/top_feature_report_example.png">
</p>

The corresponding shap values will be stored in `./results/shap_group_by_measurment.csv` and `./results/shap_group_by_timeslot.csv`.

## Other applications of this method

This method can be generalized to be used on other hospitalization data. One application of this method is the [COVID-19 DREAM Challenge](https://www.synapse.org/#!Synapse:syn21849255/wiki/602411), where this method also achieves top performance.

## Reference
* For citation, please refer to our latest iScience paper: [Assessment of the timeliness and robustness for predicting adult sepsis](https://www.sciencedirect.com/science/article/pii/S2589004221000742).
* For protocol(TBD)
