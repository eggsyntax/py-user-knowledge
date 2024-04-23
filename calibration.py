import pickle 
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression

class IsotonicCalibrator:
    def __init__(self):
        self.isotonic_models = None

    def fit(self, ground_truth, predictions):
        # Get the number of classes
        num_classes = ground_truth.shape[1]

        # Create an isotonic regression model for each class
        self.isotonic_models = [IsotonicRegression(out_of_bounds='clip') for _ in range(num_classes)]

        # Fit the isotonic regression model for each class
        for i in range(num_classes):
            self.isotonic_models[i].fit(predictions[:, i], ground_truth[:, i])

    def calibrate(self, predictions):
        # Check if the isotonic models have been fitted
        if self.isotonic_models is None:
            raise ValueError("Calibrator has not been fitted yet. Call 'fit' before 'calibrate'.")

        # Calibrate the predictions using the fitted isotonic regression models
        calibrated_predictions = np.zeros_like(predictions)
        for i in range(len(self.isotonic_models)):
            calibrated_predictions[:, i] = self.isotonic_models[i].predict(predictions[:, i])

        # Normalize the calibrated predictions to ensure they sum to 1 for each case
        calibrated_predictions /= np.sum(calibrated_predictions, axis=1, keepdims=True)

        return calibrated_predictions

    def save(self, file_path):
        # Create directory if needed
        file = Path(file_path)
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path):
        try:
            with open(file_path, 'rb') as file:
                return pickle.load(file)
        except:
            print(f"No saved calibrator at {file_path}")

def fitted_isotonic_calibrator(ground_truth, guesses):
    """Fit an isotonic calibrator to the guesses."""
    calibrator = IsotonicCalibrator()
    calibrator.fit(ground_truth, guesses)
    return calibrator

def get_file_path(dir, category):
    return f"{dir}/calibration/{category}_calibrator.pkl"

def run_example():
    # Example usage
    ground_truth = np.array([[1, 0], [1, 0], [1, 0]])
    predictions = np.array([[0.89, 0.11], [0.3, 0.7], [0.68, 0.32]])

    # Create an instance of the IsotonicCalibrator
    calibrator = fitted_isotonic_calibrator(ground_truth, predictions)

    calibrated_current_predictions = calibrator.calibrate(predictions)
    # Print the calibrated current predictions
    print("Calibrated Current Predictions:")
    print(calibrated_current_predictions)

    # Future predictions
    future_predictions = np.array([[0.92, 0.08], [0.75, 0.25], [0.6, 0.4]])

    # Calibrate the future predictions using the fitted calibrator
    calibrated_future_predictions = calibrator.calibrate(future_predictions)

    # Print the calibrated future predictions
    print("Calibrated Future Predictions:")
    print(calibrated_future_predictions)
