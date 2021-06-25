## Predictive Performance 

**Files description**

_task_ and _worker_: distributed system to perform the learning tasks.

_PredictivePerformance.py_: build classifiers and evaluate them.

_ConcatPerformance_: generate csv files with the concatenated results - 5 * 5 CV - for each learning algorithm.

_PerformanceAnalysis_: analyse the averaged results obtained after modeling. 

_StatisticalTests_: apply Benavoli test to the results. 

**Execution**

Run the following command to put on the queue the files that will be evaluated.
```
python3 Modeling/task.py  --input_folder "Data/Final_results/AllinputFiles/Originals"
```

The next command is for running the evaluation.
```
python3 Modeling/worker.py  --input_folder "Data/Final_results/AllinputFiles" --output_folder "Data/Final_results/AlloutputFiles/Originals"
```

To average the performance, just run the script.
```
python3 Modeling/AvgPerformance.py
```