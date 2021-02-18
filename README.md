# sepsis
##Top-performing algorithm for DII National Data Science Challenge

The example data in the data/ are randomly generated data for the demonstration of the algorithm

## Input data format

Two types of data is requied for model training and prediction:
`gs.file`: `.txt` file two columns. The first column if file name index. The second column is the gold standard (0/1), representing the final outbreak of sepsis

``` r
0.psv,1
1.psv,1
2.psv,0
3.psv,1
4.psv,0
5.psv,1
6.psv,1
``` 

`*.psv`: `.psv` table file separated by '|', which is the time-series feature records.
	The header of psv file are the feature names. To note, the first column is the time index.

``` r
HR feature_1 featuyre_2 ... feature_n-1 feature_n
0.0 1 0.0 ... 1.3 0.0 
1.0 NaN 0.0 ... 0.0 0.0
3.5 NaN 2.3 ... 0.0 0.0
```
## model training and cross validation
``` r
python main.py -g [PATH]
```

## dependency



