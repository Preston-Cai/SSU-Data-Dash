# Saved validation error analysis

V8 is the reference. Current means the saved output file, not the current source code.
Text associations are exploratory and may reflect property mix; they are not causal effects.

## v8

### track

```
                n  MdAPE   MAPE  median_signed  p90_APE
track                                                  
list_aware  10069   3.98   5.08          -0.28    10.62
off_market  10069   8.14  10.74           0.31    22.62
```

### house_type_name

```
                                  n  MdAPE   MAPE  median_signed  p90_APE
track      house_type_name                                               
list_aware Co-Op Apt             48   4.25   5.09           1.20     9.92
           Condo Apt           4843   3.24   3.86          -0.23     7.87
           Condo Townhouse      616   3.86   4.79           0.08    10.18
           Detached            3123   4.99   6.27           0.04    13.42
           Freehold Townhouse   405   5.10   6.51          -0.83    13.59
           Semi-Detached       1034   5.68   6.78          -1.83    14.17
off_market Co-Op Apt             48  12.43  19.55           2.04    34.55
           Condo Apt           4843   7.07   9.39           1.25    19.50
           Condo Townhouse      616   6.48   8.29           0.64    17.63
           Detached            3123  10.23  12.88          -0.30    26.93
           Freehold Townhouse   405   9.62  11.51          -2.72    23.19
           Semi-Detached       1034   9.62  11.35          -3.38    22.70
```

### month

```
                       n  MdAPE   MAPE  median_signed  p90_APE
track      month                                              
list_aware 2025-12  1238   3.64   4.69           0.50    10.08
           2026-01  1058   3.89   4.78          -0.03     9.75
           2026-02  1442   3.91   5.11          -1.18    10.56
           2026-03  1816   4.04   5.21          -1.03    10.71
           2026-04  2224   3.91   5.01          -0.07    10.32
           2026-05  2291   4.31   5.37           0.07    11.33
off_market 2025-12  1238   8.23  10.66           1.92    23.08
           2026-01  1058   8.21  10.74          -0.40    22.47
           2026-02  1442   8.50  10.62          -2.30    22.03
           2026-03  1816   7.84  10.50          -0.77    22.63
           2026-04  2224   7.95  10.69           1.39    22.38
           2026-05  2291   8.23  11.10           1.25    23.13
```

### size_band

```
                                n  MdAPE   MAPE  median_signed  p90_APE
track      size_band                                                   
list_aware (0.0, 700.0]      1651   3.31   3.87          -0.32     7.77
           (700.0, 1200.0]   3662   3.49   4.42          -0.12     9.14
           (1200.0, 1800.0]  2685   4.64   5.72          -0.51    12.04
           (1800.0, 2500.0]  1170   5.00   6.05          -0.38    12.20
           (2500.0, inf]      901   5.26   6.77          -0.18    14.69
off_market (0.0, 700.0]      1651   6.33   7.73           1.38    16.12
           (700.0, 1200.0]   3662   7.20   8.97           0.81    18.87
           (1200.0, 1800.0]  2685   9.48  12.10          -1.02    24.95
           (1800.0, 2500.0]  1170   9.64  12.69          -1.56    27.14
           (2500.0, inf]      901  13.21  16.85           0.80    34.83
```

### price_band

```
                                       n  MdAPE   MAPE  median_signed  p90_APE
track      price_band                                                         
list_aware (226199.999, 521000.0]   2015   3.34   3.99           0.72     7.97
           (521000.0, 692000.0]     2013   3.38   4.13          -0.09     8.53
           (692000.0, 939300.0]     2015   3.93   4.92          -0.24    10.33
           (939300.0, 1316600.0]    2012   4.42   5.68          -0.85    12.11
           (1316600.0, 17035700.0]  2014   5.38   6.67          -1.44    14.19
off_market (226199.999, 521000.0]   2015   6.76   9.11           4.49    18.80
           (521000.0, 692000.0]     2013   6.79   8.85           1.56    18.61
           (692000.0, 939300.0]     2015   7.66   9.99           0.85    20.68
           (939300.0, 1316600.0]    2012   8.92  10.72          -2.63    21.59
           (1316600.0, 17035700.0]  2014  12.34  15.04          -6.70    30.65
```

### municipality

```
                            n  MdAPE   MAPE  median_signed  p90_APE
track      municipality                                            
list_aware Etobicoke      971   3.75   4.80           0.00    10.05
           North York    1798   3.90   4.96          -0.38    10.30
           Scarborough   1430   4.02   4.99          -0.37    10.35
           Toronto       5746   4.04   5.19          -0.24    10.84
           York           124   4.09   4.79          -0.67     9.77
off_market Etobicoke      971   7.80  10.52           0.26    22.42
           North York    1798   7.91  10.50          -0.45    22.11
           Scarborough   1430   7.60   9.88           0.21    21.33
           Toronto       5746   8.50  11.12           0.56    23.33
           York           124   6.45   8.54           1.41    17.93
```

### community

```
                             n  MdAPE   MAPE  median_signed  p90_APE
track      community                                                
list_aware Ashworth        150   3.32   4.48          -0.86     9.63
           Balsamgrove     115   3.72   4.46           1.13     8.62
           Beechbourne     188   3.88   4.79          -0.20    10.09
           Cedarbank       101   4.28   5.83          -0.75    12.40
           Cedarglen       102   3.11   3.85          -0.12     8.10
           Cedarview       208   3.27   4.00          -0.10     8.09
           Cedarwood       113   3.83   5.45           0.88     9.66
           Chestnutbank    100   3.59   4.21           0.21     8.28
           Chestnutbourne  122   4.02   4.87           0.36    10.67
           Cypressdale     140   4.18   5.21          -0.87     9.91
           Cypresspark     124   3.46   4.03          -0.60     8.34
           Elmgardens      101   3.80   4.54          -1.01     9.06
           Elmview         184   3.66   4.20          -0.03     8.24
           Hawthornbourne  120   3.99   4.60          -0.16     8.84
           Hemlockvale     218   5.04   6.35          -0.21    14.94
           Hickorybourne   157   3.36   4.05           0.09     8.10
           Hickoryhaven    243   3.51   4.41           0.50     8.69
           Junipergardens  196   4.69   5.67          -0.31    12.19
           Lindendale      101   4.94   5.53           0.16    10.05
           Lindenridge     119   3.07   3.98          -1.23     7.73
           Oakbourne       122   3.68   4.07           1.21     7.44
           Rowanbrook      122   4.25   5.31          -0.03    11.25
           Rowangardens    134   4.19   4.95          -1.09     9.32
           Spruceview      132   3.77   4.39           0.25     9.87
           Sumacpark       114   3.45   4.17          -0.77     8.75
           Sumacwood       102   3.68   4.01          -0.54     7.78
           Walnutfield     127   2.91   3.97           0.19     9.33
           Willowglen      222   4.22   5.56          -1.71    12.26
off_market Ashworth        150   8.30  10.27           0.23    20.79
           Balsamgrove     115   8.16   9.80           2.02    19.60
           Beechbourne     188   8.73  10.41          -0.16    20.30
           Cedarbank       101   9.02  11.51          -1.67    24.48
           Cedarglen       102   6.42   8.32           2.92    18.13
           Cedarview       208   6.94   8.83           0.14    19.04
           Cedarwood       113   6.98   8.57           0.38    17.49
           Chestnutbank    100   7.61  10.39           0.08    22.57
           Chestnutbourne  122   7.10   9.19          -0.88    18.28
           Cypressdale     140   9.48  12.76           0.53    25.29
           Cypresspark     124   7.63  10.25           1.04    22.02
           Elmgardens      101   8.29   9.72           1.04    18.97
           Elmview         184  10.79  15.33          -0.60    33.34
           Hawthornbourne  120  12.43  14.52          -1.40    30.06
           Hemlockvale     218   8.93  13.10           0.81    28.55
           Hickorybourne   157   6.53   7.95          -0.31    16.55
           Hickoryhaven    243   6.56   8.92           2.54    18.97
           Junipergardens  196   9.32  12.56           0.07    27.52
           Lindendale      101   8.72  13.09           1.12    30.12
           Lindenridge     119   7.59  10.93          -0.30    19.48
           Oakbourne       122   6.50   8.81           3.59    16.93
           Rowanbrook      122   7.79  10.80           2.39    23.17
           Rowangardens    134   6.82   8.67           0.08    19.92
           Spruceview      132   7.27   9.36          -0.70    19.26
           Sumacpark       114   5.92   7.97           0.25    17.24
           Sumacwood       102   8.17  12.40           4.90    27.30
           Walnutfield     127   8.22  10.50           1.69    20.46
           Willowglen      222   7.41   9.39          -2.30    20.73
```

### Text associations, off-market

```
             phrase   n  mean_log_actual_over_prediction_pct  Dec-Feb  Mar-May
           new roof 138                                 9.30     7.42    10.15
 efficiency furnace 141                                 9.02     7.87     9.56
    high efficiency 141                                 9.02     7.87     9.56
          roof high 137                                 8.84     7.38     9.51
               high 161                                 8.26     6.57     9.16
         landscaped 160                                 8.13     9.26     7.33
 landscaped gardens 132                                 7.46     8.90     6.39
       new flooring 128                                -7.10    -6.58    -7.44
                rec 160                                 7.09     6.36     7.40
         room extra 128                                 7.05     4.72     7.96
       concept main 169                                 7.03     7.83     6.59
           rec room 155                                 7.03     6.16     7.38
         efficiency 182                                 7.00     5.34     7.86
     extra bathroom 121                                 6.87     5.03     7.62
    smooth ceilings 341                                 6.83     5.90     7.25
           ceilings 344                                 6.79     5.90     7.19
        new central 143                                 6.77     7.13     6.61
             smooth 353                                 6.74     5.59     7.29
         completely 175                                 6.72     7.74     6.16
  completely redone 113                                 6.67     8.31     5.70
   built appliances 350                                 6.39     8.35     5.01
       good ceiling 102                                 6.17     6.14     6.19
 updated electrical 272                                 6.16     6.08     6.22
electrical plumbing 354                                 6.12     6.77     5.73
           plumbing 356                                 6.04     6.57     5.73
              built 384                                 5.97     8.21     4.47
             luxury 162                                 5.97     5.49     6.20
         redone new 104                                 5.91     6.37     5.67
           stunning 265                                 5.76     6.17     5.53
         pot lights 361                                 5.76     8.16     4.24
```

## current saved run

### track

```
                n  MdAPE   MAPE  median_signed  p90_APE
track                                                  
list_aware  10069   3.98   5.08          -0.28    10.62
off_market  10069   7.95  10.31           0.18    21.75
```

### house_type_name

```
                                  n  MdAPE   MAPE  median_signed  p90_APE
track      house_type_name                                               
list_aware Co-Op Apt             48   4.25   5.09           1.20     9.92
           Condo Apt           4843   3.24   3.86          -0.23     7.87
           Condo Townhouse      616   3.86   4.79           0.08    10.18
           Detached            3123   4.99   6.27           0.04    13.42
           Freehold Townhouse   405   5.10   6.51          -0.83    13.59
           Semi-Detached       1034   5.68   6.78          -1.83    14.17
off_market Co-Op Apt             48  14.56  19.44           1.65    35.80
           Condo Apt           4843   6.87   9.01           1.32    18.83
           Condo Townhouse      616   6.64   8.09          -0.07    16.80
           Detached            3123   9.91  12.29          -0.31    24.97
           Freehold Townhouse   405   9.81  11.51          -2.72    23.67
           Semi-Detached       1034   9.05  10.85          -3.02    22.72
```

### month

```
                       n  MdAPE   MAPE  median_signed  p90_APE
track      month                                              
list_aware 2025-12  1238   3.64   4.69           0.50    10.08
           2026-01  1058   3.89   4.78          -0.03     9.75
           2026-02  1442   3.91   5.11          -1.18    10.56
           2026-03  1816   4.04   5.21          -1.03    10.71
           2026-04  2224   3.91   5.01          -0.07    10.32
           2026-05  2291   4.31   5.37           0.07    11.33
off_market 2025-12  1238   8.19  10.46           2.84    22.24
           2026-01  1058   7.62  10.16           0.05    21.29
           2026-02  1442   8.20  10.13          -2.80    20.48
           2026-03  1816   7.65  10.03          -1.13    20.79
           2026-04  2224   8.01  10.22           1.01    21.73
           2026-05  2291   7.98  10.74           1.25    22.90
```

### size_band

```
                                n  MdAPE   MAPE  median_signed  p90_APE
track      size_band                                                   
list_aware (0.0, 700.0]      1651   3.31   3.87          -0.32     7.77
           (700.0, 1200.0]   3662   3.49   4.42          -0.12     9.14
           (1200.0, 1800.0]  2685   4.64   5.72          -0.51    12.04
           (1800.0, 2500.0]  1170   5.00   6.05          -0.38    12.20
           (2500.0, inf]      901   5.26   6.77          -0.18    14.69
off_market (0.0, 700.0]      1651   6.17   7.63           1.65    16.23
           (700.0, 1200.0]   3662   7.03   8.70           0.81    18.11
           (1200.0, 1800.0]  2685   9.12  11.48          -1.38    23.95
           (1800.0, 2500.0]  1170   9.88  12.23          -1.81    24.56
           (2500.0, inf]      901  12.58  15.80           0.54    31.95
```

### price_band

```
                                       n  MdAPE   MAPE  median_signed  p90_APE
track      price_band                                                         
list_aware (226199.999, 521000.0]   2015   3.34   3.99           0.72     7.97
           (521000.0, 692000.0]     2013   3.38   4.13          -0.09     8.53
           (692000.0, 939300.0]     2015   3.93   4.92          -0.24    10.33
           (939300.0, 1316600.0]    2012   4.42   5.68          -0.85    12.11
           (1316600.0, 17035700.0]  2014   5.38   6.67          -1.44    14.19
off_market (226199.999, 521000.0]   2015   6.72   8.91           4.18    18.16
           (521000.0, 692000.0]     2013   6.54   8.57           1.68    18.29
           (692000.0, 939300.0]     2015   7.40   9.55           0.65    20.18
           (939300.0, 1316600.0]    2012   8.53  10.27          -2.70    20.42
           (1316600.0, 17035700.0]  2014  11.94  14.27          -6.56    28.84
```

### municipality

```
                            n  MdAPE   MAPE  median_signed  p90_APE
track      municipality                                            
list_aware Etobicoke      971   3.75   4.80           0.00    10.05
           North York    1798   3.90   4.96          -0.38    10.30
           Scarborough   1430   4.02   4.99          -0.37    10.35
           Toronto       5746   4.04   5.19          -0.24    10.84
           York           124   4.09   4.79          -0.67     9.77
off_market Etobicoke      971   7.77  10.27           0.49    21.68
           North York    1798   7.51   9.92          -0.52    20.73
           Scarborough   1430   7.49   9.46          -0.38    19.73
           Toronto       5746   8.28  10.70           0.45    22.70
           York           124   6.77   8.32           1.45    16.62
```

### community

```
                             n  MdAPE   MAPE  median_signed  p90_APE
track      community                                                
list_aware Ashworth        150   3.32   4.48          -0.86     9.63
           Balsamgrove     115   3.72   4.46           1.13     8.62
           Beechbourne     188   3.88   4.79          -0.20    10.09
           Cedarbank       101   4.28   5.83          -0.75    12.40
           Cedarglen       102   3.11   3.85          -0.12     8.10
           Cedarview       208   3.27   4.00          -0.10     8.09
           Cedarwood       113   3.83   5.45           0.88     9.66
           Chestnutbank    100   3.59   4.21           0.21     8.28
           Chestnutbourne  122   4.02   4.87           0.36    10.67
           Cypressdale     140   4.18   5.21          -0.87     9.91
           Cypresspark     124   3.46   4.03          -0.60     8.34
           Elmgardens      101   3.80   4.54          -1.01     9.06
           Elmview         184   3.66   4.20          -0.03     8.24
           Hawthornbourne  120   3.99   4.60          -0.16     8.84
           Hemlockvale     218   5.04   6.35          -0.21    14.94
           Hickorybourne   157   3.36   4.05           0.09     8.10
           Hickoryhaven    243   3.51   4.41           0.50     8.69
           Junipergardens  196   4.69   5.67          -0.31    12.19
           Lindendale      101   4.94   5.53           0.16    10.05
           Lindenridge     119   3.07   3.98          -1.23     7.73
           Oakbourne       122   3.68   4.07           1.21     7.44
           Rowanbrook      122   4.25   5.31          -0.03    11.25
           Rowangardens    134   4.19   4.95          -1.09     9.32
           Spruceview      132   3.77   4.39           0.25     9.87
           Sumacpark       114   3.45   4.17          -0.77     8.75
           Sumacwood       102   3.68   4.01          -0.54     7.78
           Walnutfield     127   2.91   3.97           0.19     9.33
           Willowglen      222   4.22   5.56          -1.71    12.26
off_market Ashworth        150   7.17   9.94          -0.45    20.26
           Balsamgrove     115   8.48   9.59           2.69    18.13
           Beechbourne     188   8.78  10.52          -0.52    20.79
           Cedarbank       101   9.95  11.85          -1.66    24.71
           Cedarglen       102   5.82   7.14           1.73    15.02
           Cedarview       208   6.96   8.60           0.54    17.60
           Cedarwood       113   6.98   8.14           1.20    17.17
           Chestnutbank    100   7.06  10.01          -0.23    19.34
           Chestnutbourne  122   5.87   7.89          -1.05    16.15
           Cypressdale     140   9.30  11.77          -0.09    22.99
           Cypresspark     124   7.38   9.91          -0.17    21.31
           Elmgardens      101   8.36   9.62           1.64    17.66
           Elmview         184  10.20  14.35          -0.61    31.18
           Hawthornbourne  120  11.35  12.95          -0.93    24.67
           Hemlockvale     218   9.81  13.24           1.58    27.17
           Hickorybourne   157   7.05   8.04          -0.29    16.79
           Hickoryhaven    243   7.06   8.82           2.38    17.50
           Junipergardens  196   8.93  11.11           1.07    24.65
           Lindendale      101   8.98  12.08           1.19    26.07
           Lindenridge     119   6.46   9.63          -1.47    19.37
           Oakbourne       122   6.02   8.21           2.14    16.82
           Rowanbrook      122   7.58  10.68           2.09    23.36
           Rowangardens    134   6.97   8.13           0.07    16.49
           Spruceview      132   6.81   8.44          -1.81    18.39
           Sumacpark       114   5.73   7.54           0.17    15.22
           Sumacwood       102   8.34  11.98           5.07    24.56
           Walnutfield     127   7.33   9.60           0.75    20.55
           Willowglen      222   7.16   9.10          -2.53    20.10
```
