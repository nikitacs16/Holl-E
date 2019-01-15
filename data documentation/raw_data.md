This folder contains the pre-processed data before it is converted to the respective baselines.
For converting this data to respective baselines, copy the convert_as_expt.py in the same location where the raw_data folder is stored. Then run the following command

```
python convert_as_expt.py experiment_name data_type query_type 
```

Choices for experiment_name : squad, hred, gttp
Choices for data_type : oracle, oracle_reduced, full, full_reduced, mixed
Choices for query_type : query, context

For all the experiments in the paper, query_type was context, data_type mapping is as follows: 

experiment in the paper | key
------------ | -------------
oracle | oracle_reduced
mixed-long | full_reduced
mixed-short | mixed

Meanining of the keys are as follows : 

