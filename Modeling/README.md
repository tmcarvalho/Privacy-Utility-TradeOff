**Files description**

_PredictivePerformance.py_: build classifiers and evaluate them.

_task_ and _worker_: distributed system to perform the learning task.

Run the following command to put on the queue the files that will be evaluated.
```
python3 Modeling/task.py  --input_folder "Data/Final_results/AllinputFiles"
```

The next command is to execute the evaluation.
```
python3 Modeling/worker.py  --input_folder "Data/Final_results/AllinputFiles" --output_folder "Data/Final_results/AlloutputFiles"
```