# Training:
Place the txt document containing the dataset paths (provided in the data directory) into the dataset directory.
Run the train1.py file, where you can set the core parameters.

# Testing:
In testsave.py, set the datasettype parameter for the dataset to "test" for cityscape and "query" for tokyo247, then run the script.

# Calculating Metrics:
After generating the test images, use the cal_ssim() and cal_psnr() functions in testsava to calculate the metrics.