# WPdx+ data audit

## 1. Overview

```
           rows  unknown  known  no_water  report_yr_min  report_yr_max  modal_yr_share  n_sources  top_source_share
country                                                                                                             
Ethiopia  24488      597  23891     0.105           1998           2022           0.544         22             0.560
Kenya     21953      277  21676     0.088           2011           2024           0.305         15             0.492
Tanzania  17518        0  17518     0.434           1978           2024           0.332         11             0.620
Uganda    98767     2523  96244     0.172           2005           2025           0.729         26             0.801
```
is_latest values: {'True': 162726}; wpdx_id unique: True

## 2. Column profile (missing share on labelled rows, per country)

```
country                     nunique  Ethiopia  Kenya  Tanzania  Uganda
row_id                       162726     0.000  0.000     0.000   0.000
source                           62     0.000  0.000     0.000   0.000
lat_deg                      160268     0.000  0.000     0.000   0.000
lon_deg                      161731     0.000  0.000     0.000   0.000
report_date                    3367     0.000  0.000     0.000   0.000
status_id                         3     0.000  0.000     0.000   0.000
water_source_clean               11     0.007  0.006     0.000   0.154
water_source_category             7     0.007  0.006     0.000   0.154
water_tech_clean                 20     0.255  0.267     0.057   0.465
_water_tech_category              4     0.274  0.299     0.059   0.465
facility_type                     2     0.000  0.000     0.000   0.000
clean_country_name                4     0.000  0.000     0.000   0.000
clean_country_id                  4     0.000  0.000     0.000   0.000
clean_adm1                       78     0.000  0.000     0.000   0.000
clean_adm2                      458     0.000  0.000     0.000   0.000
clean_adm3                     2337     0.000  0.000     0.000   0.000
clean_adm4                     1410     1.000  1.000     1.000   0.000
install_year                    112     0.207  0.535     0.055   0.113
installer                       706     0.787  0.826     0.018   0.962
rehab_year                       14     0.997  1.000     1.000   0.999
rehabilitator                     1     0.999  1.000     1.000   1.000
management_clean                  9     0.239  0.537     0.023   0.094
status_clean                      7     0.000  0.000     0.000   0.000
pay_clean                         8     0.908  0.835     0.006   0.639
fecal_coliform_presence           2     0.985  1.000     1.000   0.984
fecal_coliform_value             88     0.985  1.000     1.000   0.995
subjective_quality_clean          5     0.999  0.532     0.013   0.968
activity_id                  111913     0.788  0.100     0.996   0.090
scheme_id                      6914     0.778  1.000     0.625   0.873
wpdx_id                      162726     0.000  0.000     0.000   0.000
notes                         35231     0.249  0.851     0.008   0.907
orig_lnk                         71     0.437  0.833     1.000   0.986
photo_lnk                     18113     0.871  0.532     0.995   0.946
country_id                        7     0.001  0.001     0.000   0.014
data_lnk                         59     0.000  0.000     0.000   0.000
distance_to_primary          162162     0.000  0.000     0.000   0.000
distance_to_secondary        162162     0.000  0.000     0.000   0.000
distance_to_tertiary         162162     0.000  0.000     0.000   0.000
distance_to_city             162106     0.000  0.002     0.000   0.000
distance_to_town             162162     0.000  0.000     0.000   0.000
water_point_history           32030     0.000  0.000     0.000   0.000
local_population              10302     0.010  0.040     0.025   0.175
served_population              4490     0.010  0.040     0.025   0.175
rehab_priority                 3173     0.895  0.918     0.584   0.861
_pop_who_would_gain_access    11336     0.896  0.918     0.582   0.839
crucialness                  107249     0.027  0.044     0.029   0.197
pressure                       6752     0.027  0.086     0.029   0.198
usage_cap                         4     0.008  0.049     0.000   0.146
days_since_report              3367     0.000  0.000     0.000   0.000
staleness_score                3367     0.000  0.000     0.000   0.000
is_latest                         1     0.000  0.000     0.000   0.000
location_id                  162726     0.000  0.000     0.000   0.000
is_urban                          2     0.000  0.000     0.000   0.000
cluster_size                     72     0.000  0.000     0.000   0.000
country_name                      4     0.001  0.001     0.000   0.014
water_source                    264     0.040  0.003     0.001   0.016
water_tech                      387     0.745  0.510     0.005   0.902
status                         3695     0.953  0.524     0.005   0.087
adm1                            276     0.031  0.119     0.005   0.054
adm2                            939     0.212  0.238     0.005   0.036
adm3                           1789     0.182  0.268     0.999   0.913
management                      198     0.236  0.535     0.011   0.090
pay                             199     0.908  0.899     0.006   0.671
subjective_quality              322     0.999  0.530     0.011   0.965
new_georeferenced_column_    162726     0.000  0.000     0.000   0.000
lat_lon_deg                  162724     0.000  0.000     0.000   0.000
public_data_source               59     0.000  0.000     0.000   0.000
converted                        18     0.860  0.523     0.001   0.111
count                             1     0.000  0.000     0.000   0.000
created_timestamp                59     0.000  0.000     0.000   0.000
updated                          59     0.000  0.000     0.000   0.000
is_duplicate                      1     0.989  0.969     0.996   0.991
dataset_title                    59     0.000  0.000     0.000   0.000
```

Missing >95% in every country: ['rehab_year', 'rehabilitator', 'fecal_coliform_presence', 'fecal_coliform_value', 'is_duplicate']
Constant columns: ['rehabilitator', 'is_latest', 'count', 'is_duplicate']

## 3. Single-feature predictive power for 'no water' (leakage / fingerprint scan)

Numeric: |AUC-0.5|+0.5 of the raw value. Categorical: AUC of 5-fold out-of-fold target encoding (NaN is its own category). ALL = pooled, so country base-rate differences inflate pooled scores.
```
                           kind  nunique  coverage   ALL  Ethiopia  Kenya  Tanzania  Uganda
column                                                                                     
water_point_history         cat    31050     1.000 0.986     0.979  0.786     0.995   0.996
status                      cat     3668     0.732 0.961     0.705  0.916     0.717   0.988
status_clean                cat        7     1.000 0.831     0.592  0.720     0.796   0.936
adm2                        cat      889     0.914 0.755     0.852  0.766     0.708   0.678
report_date                 cat     3298     1.000 0.748     0.845  0.767     0.740   0.649
clean_adm3                  cat     2302     1.000 0.748     0.840  0.738     0.784   0.652
clean_adm2                  cat      451     1.000 0.737     0.763  0.750     0.712   0.650
adm1                        cat      273     0.946 0.734     0.655  0.707     0.664   0.672
water_source                cat      258     0.984 0.719     0.848  0.773     0.592   0.601
source                      cat       61     1.000 0.712     0.853  0.747     0.547   0.580
dataset_title               cat       58     1.000 0.712     0.853  0.753     0.499   0.580
public_data_source          cat       58     1.000 0.712     0.853  0.753     0.499   0.580
clean_adm1                  cat       78     1.000 0.695     0.598  0.719     0.665   0.585
management                  cat      192     0.836 0.667     0.788  0.710     0.569   0.575
converted                   cat       18     0.733 0.662     0.600  0.693     0.503   0.534
report_year                 num       26     1.000 0.657     0.594  0.650     0.536   0.546
install_valid               num      106     0.822 0.655     0.526  0.507     0.679   0.594
days_since_report           num     3298     1.000 0.652     0.551  0.643     0.534   0.537
staleness_score             num     3298     1.000 0.652     0.551  0.643     0.534   0.537
country_id                  cat        7     0.991 0.643     0.484  0.488       NaN   0.506
country_name                cat        4     0.991 0.642       NaN    NaN       NaN     NaN
pay                         cat      199     0.336 0.641     0.612  0.524     0.631   0.588
pay_clean                   cat        8     0.364 0.638     0.604  0.528     0.629   0.594
clean_country_id            cat        4     1.000 0.638       NaN    NaN       NaN     NaN
[country]                   cat        4     1.000 0.638       NaN    NaN       NaN     NaN
water_tech                  cat      383     0.274 0.634     0.690  0.718     0.673   0.527
clean_adm4                  cat     1409     0.604 0.625       NaN    NaN       NaN   0.681
installer                   cat      701     0.187 0.624     0.637  0.612     0.708   0.516
crucialness                 num   104817     0.868 0.618     0.508  0.659     0.719   0.536
fecal_coliform_value        num       87     0.005 0.617     0.563    NaN       NaN   0.576
subjective_quality          cat      321     0.194 0.616     0.484  0.721     0.559   0.517
water_tech_clean            cat       20     0.638 0.613     0.641  0.711     0.593   0.545
age                         num       96     0.822 0.604     0.505  0.549     0.676   0.605
adm3                        cat     1785     0.275 0.601     0.814  0.738     0.489   0.526
subjective_quality_clean    cat        5     0.192 0.594       NaN  0.661     0.543   0.507
distance_to_city            num   158721     1.000 0.584     0.506  0.612     0.514   0.550
water_source_clean          cat       11     0.905 0.582     0.582  0.715     0.564   0.533
management_clean            cat        9     0.832 0.581     0.691  0.692     0.526   0.561
_water_tech_category        cat        4     0.631 0.579     0.490  0.702     0.533   0.539
scheme_id                   cat     6838     0.151 0.573     0.646    NaN     0.645   0.552
pressure                    num     6613     0.861 0.572     0.500  0.553     0.668   0.523
served_population           num     4378     0.885 0.566     0.511  0.523     0.676   0.515
distance_to_town            num   158777     1.000 0.562     0.537  0.633     0.526   0.519
local_population            num    10136     0.885 0.534     0.515  0.660     0.519   0.511
water_source_category       cat        7     0.905 0.530     0.545  0.550     0.550   0.522
distance_to_tertiary        num   158777     1.000 0.526     0.502  0.536     0.520   0.513
distance_to_secondary       num   158777     1.000 0.515     0.672  0.506     0.515   0.500
usage_cap                   num        4     0.904 0.509     0.537  0.531     0.559   0.509
distance_to_primary         num   158777     1.000 0.509     0.534  0.564     0.525   0.504
cluster_size                num       72     1.000 0.508     0.515  0.502     0.502   0.507
fecal_coliform_presence     cat        2     0.012 0.503     0.492    NaN       NaN   0.505
is_urban                    cat        2     1.000 0.501     0.486  0.493     0.489   0.502
facility_type               cat        2     1.000 0.496     0.484  0.487     0.488   0.496
rehab_year                  num       14     0.001   NaN       NaN    NaN       NaN     NaN
rehabilitator               cat        1     0.000   NaN       NaN    NaN       NaN     NaN
rehab_priority              num     2940     0.157   NaN       NaN    NaN       NaN     NaN
_pop_who_would_gain_access  num    11336     0.170   NaN       NaN    NaN       NaN     NaN
is_latest                   cat        1     1.000   NaN       NaN    NaN       NaN     NaN
count                       num        1     1.000   NaN       NaN    NaN       NaN     NaN
is_duplicate                cat        1     0.011   NaN       NaN    NaN       NaN     NaN
```

Columns with >=0.85 strength in any country (leakage candidates): ['water_point_history', 'status', 'status_clean', 'adm2', 'source', 'dataset_title', 'public_data_source']

## 4. Is 'being missing' itself predictive? (per country, columns with 1-99% missing)

```
                            Eth_miss  Ken_miss  Tan_miss  Uga_miss  best  Uga_str  Eth_str  Ken_str  Tan_str
column                                                                                                      
rehab_priority                 0.895     0.918     0.584     0.861 0.999    0.905    0.999    0.964    0.979
_pop_who_would_gain_access     0.896     0.918     0.582     0.839 0.996    0.968    0.996    0.965    0.981
status                         0.953     0.524     0.005     0.087 0.716    0.528    0.716    0.676      NaN
water_tech                     0.745     0.510     0.005     0.902 0.683    0.510    0.656    0.683      NaN
management_clean               0.239     0.537     0.023     0.094 0.681    0.500    0.576    0.681    0.504
management                     0.236     0.535     0.011     0.090 0.679    0.502    0.573    0.679    0.504
converted                      0.860     0.523     0.001     0.111 0.676    0.520    0.578    0.676      NaN
subjective_quality             0.999     0.530     0.011     0.965 0.675    0.513      NaN    0.675    0.504
install_year                   0.207     0.535     0.055     0.113 0.671    0.501    0.628    0.671    0.512
subjective_quality_clean       0.999     0.532     0.013     0.968 0.664    0.510      NaN    0.664    0.504
activity_id                    0.788     0.100     0.996     0.090 0.658    0.503    0.658    0.522      NaN
adm3                           0.182     0.268     0.999     0.913 0.654    0.519    0.654    0.503      NaN
scheme_id                      0.778     1.000     0.625     0.873 0.650    0.509    0.650      NaN    0.510
adm2                           0.212     0.238     0.005     0.036 0.630    0.520    0.630    0.513      NaN
installer                      0.787     0.826     0.018     0.962 0.622    0.504    0.580    0.622    0.506
pay                            0.908     0.899     0.006     0.671 0.615    0.578    0.615    0.504      NaN
pay_clean                      0.908     0.835     0.006     0.639 0.615    0.565    0.615    0.520      NaN
```

## 5. Duplicates

```
          is_duplicate_flag  same_coord_rows  same_coord_rows_flagged  coord_groups  groups_with_conflicting_status  exact_attr_dupes
country                                                                                                                              
Ethiopia                258                2                        0             1                               1                 0
Kenya                   666                0                        0             0                               0                 0
Tanzania                 62                0                        0             0                               0                 0
Uganda                  853                0                        0             0                               0                 0
```

## 6. Coordinates

```
          lat_lon_missing  outside_country_bbox  exactly_zero
country                                                      
Ethiopia                0                     0             0
Kenya                   0                     0             0
Tanzania                0                     0             0
Uganda                  0                     0             0
```

## 7. install_year: how many 'impossible' values under different rules (all rows)

```
          present  lt_1900  lt_1950  eq_0  gt_2026  gt_report_year  lt1900_or_gt_report  lt1950_or_gt_report
country                                                                                                     
Ethiopia    19448        0        1     0        1              20                   20                   21
Kenya       10166        0       56     0        1               7                    7                   63
Tanzania    16554        0       21     0        0              21                   21                   42
Uganda      86195        0      238     0        4               4                    4                  242
TOTAL      132363        0      316     0        6              52                   52                  368
```

## 8. Numeric sanity (labelled rows)

```
                                null     min    median        p99        max  negative
column                country                                                         
local_population      Ethiopia 0.010   0.000   959.000   9997.170  70614.000         0
                      Kenya    0.040   0.000  2082.000  12339.680  67783.000         0
                      Tanzania 0.025   0.000  1028.000  10086.860  41063.000         0
                      Uganda   0.175   0.000  1219.000  14510.010  53167.000         0
distance_to_primary   Ethiopia 0.000   2.953 56272.302 148830.666 202605.837         0
                      Kenya    0.000   0.509 10958.267  92813.060 208291.106         0
                      Tanzania 0.000   0.658 30588.637 229496.655 249271.072         0
                      Uganda   0.000   0.169 12798.419  70729.448 113267.725         0
distance_to_secondary Ethiopia 0.000   0.618 18821.081  85823.187 184552.167         0
                      Kenya    0.000   0.156  3872.606  24246.739 115710.941         0
                      Tanzania 0.000   0.032  7566.655  73467.916 130047.987         0
                      Uganda   0.000   0.034  3524.433  29629.278  48801.575         0
distance_to_tertiary  Ethiopia 0.000   0.015  5889.149  58889.643 130490.406         0
                      Kenya    0.000   0.112  1703.878  22381.733  86007.508         0
                      Tanzania 0.000   0.124  1604.079  30668.746  86854.555         0
                      Uganda   0.000   0.022  1593.083  16915.828  33601.773         0
distance_to_city      Ethiopia 0.000 548.796 63315.487 167348.522 320039.778         0
                      Kenya    0.002 103.650 28311.633 148365.721 347695.468         0
                      Tanzania 0.000  27.648 74413.494 191422.053 271798.880         0
                      Uganda   0.000  55.532 35888.969 147025.302 222690.455         0
distance_to_town      Ethiopia 0.000  14.408 16276.253  63682.342 163129.752         0
                      Kenya    0.000  11.694  8942.612  45667.861 112518.150         0
                      Tanzania 0.000  22.008 20970.442  94837.852 131920.504         0
                      Uganda   0.000   3.799 13979.787  46379.802  58399.951         0
usage_cap             Ethiopia 0.008  50.000   300.000    300.000    300.000         0
                      Kenya    0.049  50.000   300.000    300.000    300.000         0
                      Tanzania 0.000  50.000   300.000    300.000    300.000         0
                      Uganda   0.146  50.000   300.000    300.000    300.000         0
cluster_size          Ethiopia 0.000   1.000     1.000      2.000     17.000         0
                      Kenya    0.000   1.000     1.000      3.000     22.000         0
                      Tanzania 0.000   1.000     1.000      2.000      9.000         0
                      Uganda   0.000   1.000     1.000      3.000    175.000         0
age                   Ethiopia 0.208   0.000    13.000     23.000    120.000         0
                      Kenya    0.536   0.000     9.000     64.000    121.000         0
                      Tanzania 0.056   0.000    10.000     41.000     84.000         0
                      Uganda   0.113   0.000     6.000     50.000    114.000         0
```
is_urban values: {'False': 155932, 'True': 3397}

## 9. Categories that exist in a country but in none of the other three (unseen-category risk for model B)

```
                                n_cats  rows_with_unseen_cat                                                                    top_unseen
column                country                                                                                                             
water_source_clean    Ethiopia       9                 0.000                                                                            {}
                      Kenya         10                 0.000                                                                            {}
                      Tanzania       8                 0.000                                                         {'Packaged Water': 1}
                      Uganda         7                 0.000                                                                            {}
water_tech_clean      Ethiopia      10                 0.000                                                                            {}
                      Kenya         15                 0.005                                     {'Rope and Bucket': 81, 'Hydram Pump': 3}
                      Tanzania      12                 0.005  {'Hand Pump - Climax': 52, 'Hand Pump - Walimi': 30, 'Hand Pump - Malda': 2}
                      Uganda        14                 0.026                                  {'Kiosk': 905, 'Hand Pump - Consallen': 412}
management_clean      Ethiopia       3                 0.000                                                                            {}
                      Kenya          6                 0.000                                                                            {}
                      Tanzania       6                 0.000                                                                            {}
                      Uganda         9                 0.000                     {'Religious Institution': 27, 'Health Care Facility': 14}
pay_clean             Ethiopia       4                 0.000                                                                            {}
                      Kenya          6                 0.000                                                                            {}
                      Tanzania       6                 0.364                                                       {'Blank/unknown': 6344}
                      Uganda         6                 0.002                                           {'Fees collected – seasonally': 81}
facility_type         Ethiopia       2                 0.000                                                                            {}
                      Kenya          2                 0.000                                                                            {}
                      Tanzania       2                 0.000                                                                            {}
                      Uganda         2                 0.000                                                                            {}
water_source_category Ethiopia       6                 0.000                                                                            {}
                      Kenya          6                 0.000                                                                            {}
                      Tanzania       6                 0.000                                                         {'Packaged Water': 1}
                      Uganda         6                 0.000                                                                            {}
_water_tech_category  Ethiopia       3                 0.000                                                                            {}
                      Kenya          4                 0.005                                                       {'Rope and Bucket': 81}
                      Tanzania       3                 0.000                                                                            {}
                      Uganda         3                 0.000                                                                            {}
```

## 10. District (adm2) structure for the held-out-district design

```
          districts  min  median   max  largest_share  top5_share  districts_lt20_rows  adm2_names_in_>1_adm1  adm1_count  adm2_missing
country                                                                                                                                
Ethiopia         74    1  57.500  7060          0.296       0.636                   25                      0          12             0
Kenya           157    1  60.000  1119          0.052       0.181                   61                      1          39             0
Tanzania         89    1 178.000   843          0.048       0.183                   28                      2          23             0
Uganda          135    4 552.000  3171          0.033       0.147                    2                      0           4             0
```

## 11. Can the country be recovered from the candidate features alone?

HistGradientBoosting, 3-fold CV, labelled rows. If accuracy is far above the majority-class rate, models can 'tell which survey this is' from features, which is the confound the proposal flags.
```
                                accuracy  macro_F1  recall_Eth  recall_Ken  recall_Tan  recall_Uga
variant                                                                                           
with missing signal (NaN kept)     0.975     0.968       0.978       0.933       0.953       0.988
missing imputed away               0.960     0.948       0.966       0.850       0.943       0.986
```
Majority-class accuracy (Uganda): 0.604