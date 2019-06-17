# trafficmanagement
Traffic Management

Submission by Andy G

This is the model for the traffic management challenge. I achieved RSME of 0.025 in the validation date (80:20 train vs. validation data, no shuffling).

How to run the files on test data:
1. Download the whole folder
2. Put the test data in the same folder under the name 'demand.csv'. Make sure also that the file 'grab.h5' is in the folder because that is the trained model saved data.
3. Run runtest.py (make sure you have access to GPU as I'm using tensorflow)
4. To retrain, run retrain.py. Retraining should be around 10-15 minutes as it takes around 1 minute in the original training data per epoch. 10 epoch is optimal for my model.
