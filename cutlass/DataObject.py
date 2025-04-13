import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib
matplotlib.use('TkAgg')  # Try 'Agg', 'TkAgg', 'Qt5Agg', etc., depending on your setup
import matplotlib.pyplot as plt

# Import custom LLE classifier class
from cutlass.lle import LLEClassifier

# Import XGBoost models
from xgboost import XGBClassifier, XGBRegressor

class DataObject:
    
    def __init__(self, data=None):
        """
        Initialize the DataObject.

        :param data: A pandas DataFrame or a dictionary that can be converted into a DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = pd.DataFrame(data) if data is not None else None
        self.models = {}  # Dictionary to store trained models
        # Attributes to hold binarized data and control the active dataset.
        self.data_binarized = None  
        self.use_binarized = False  
        self.binarization_thresholds = {}  # To store critical ranges per feature

    def _get_current_data(self):
        """Return the currently active data: binarized if activated, otherwise original."""
        if self.use_binarized and self.data_binarized is not None:
            return self.data_binarized
        else:
            return self.data
    
    def display(self, num_rows=5):
        """
        Display the first few rows of the table.
        
        :param num_rows: Number of rows to display, default is 5.
        """
        print(self._get_current_data().head(num_rows))
    
    def binarize_data(self, outcome_column, inplace=False):
        """
        Create a binarized version of the dataset based on Algorithm 1 from the paper.
        For each feature (excluding the outcome column), the critical range is computed
        from the subset of rows where the outcome is True. Then each value is mapped to:
         +1 if it falls between the lower and upper thresholds (inclusive),
         -1 otherwise.
    
        :param outcome_column: The name of the column representing the outcome.
        :param inplace: If True, replace the current data with the binarized version and
                        automatically activate it. If False, store the binarized data in a
                        separate attribute.
        """

        if self.data is None:
            raise ValueError("No data available to binarize.")
        if outcome_column not in self.data.columns:
            raise ValueError(f"Outcome column '{outcome_column}' not found in data.")

        # Define the subset Z where outcome is True.
        Z = self.data[self.data[outcome_column] == True]
    
        # Compute thresholds for each feature (skip the outcome column).
        thresholds = {}
        for col in self.data.columns:
            if col == outcome_column:
                continue
            # The critical range is defined as the min and max of the feature in Z.
            thresholds[col] = (Z[col].min(), Z[col].max())
    
        # Create a copy of the original data to binarize.
        data_bin = self.data.copy()
        for col in data_bin.columns:
            if col == outcome_column:
                continue
            lower, upper = thresholds[col]
            # Transform values: +1 if within [lower, upper], otherwise -1.
            data_bin[col] = data_bin[col].apply(lambda x: 1 if lower <= x <= upper else -1)
    
        # Store the thresholds for reference.
        self.binarization_thresholds = thresholds
        if inplace:
            self.data = data_bin
            self.use_binarized = True
        else:
            self.data_binarized = data_bin

    def activate_binarized(self):
        """
        Activate the binarized dataset for subsequent operations.
        Ensure that binarized data has been created first.
        """
        if self.data_binarized is None:
            raise ValueError("Binarized data not available. Run binarize_data() first.")
        self.use_binarized = True

    def deactivate_binarized(self):
        """Switch back to using the original dataset."""
        self.use_binarized = False

    def add_column(self, column_name, data):
        """
        Add a new column to the table.

        :param column_name: Name of the new column.
        :param data: Data for the new column, should match the length of the existing DataFrame.
        """
        self.data[column_name] = data
    
    def filter_rows(self, condition):
        """
        Filter the rows based on a condition.

        :param condition: A boolean condition for filtering (e.g., self.data['column'] > value).
        :return: A new DataObject containing filtered rows.
        """
        filtered_data = self.data[condition]
        return DataObject(filtered_data)
    
    def summary_statistics(self):
        """
        Return summary statistics of the DataFrame.
        """
        return self.data.describe()
    
    def load_csv(self, file_path):
        """
        Load data from a CSV file.

        :param file_path: Path to the CSV file.
        """
        try:
            self.data = pd.read_csv(file_path)
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def save_csv(self, file_path):
        """
        Save data to a CSV file.

        :param file_path: Path to save the CSV file.
        """
        try:
            self.data.to_csv(file_path, index=False)
            return True
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return False
    
    def drop_missing(self):
        """
        Drop rows with missing values.
        """
        self.data.dropna(inplace=True)
    
    def rescale_column(self, column_name, max=1.0, min=0.0):
        """
        Rescale a column (default between 0 and 1).

        :param column_name: The name of the column to rescale.
        """
        self.data[column_name] = min + (max - min) * \
                                   (self.data[column_name] - self.data[column_name].min()) / \
                                   (self.data[column_name].max() - self.data[column_name].min())
    
    def add_random_noise(self, column_name, std_dev):
        """
        Add random noise to a column.

        :param column_name: The name of the column to add noise to.
        :param std_dev: The standard deviation of the noise.
        """
        noise = np.random.normal(0, std_dev, len(self.data))
        self.data[column_name] += noise
    
    def impute(self, cols=None, strategy="simple"):
        """
        Impute missing values in specified columns.

        :param cols: List of columns to impute. If None, all columns are considered.
        :param strategy: Imputation strategy ("simple" supported).
        """
        if cols is None:
            cols = list(self.data.columns)
        if not isinstance(cols, list):
            cols = [cols]
        
        numcols = self.data[cols].select_dtypes(include="number").columns
        catcols = [col for col in cols if col not in numcols]
        
        if strategy.upper() == "SIMPLE":
            if len(numcols) > 0:
                for cl in numcols:
                    # Impute numerical columns with mean
                    mean_imputer = SimpleImputer(strategy='mean')
                    self.data[cl] = mean_imputer.fit_transform(self.data[[cl]])
            
            if len(catcols) > 0:
                for cl in catcols:
                    # Impute categorical columns with the most frequent value (mode)
                    mode_imputer = SimpleImputer(strategy='most_frequent')
                    self.data[[cl]] = mode_imputer.fit_transform(self.data[[cl]])
    
    def plot(self, x_column, y_column, plot_type='scatter', figsize=(10, 6), title=None, xlabel=None,
             ylabel=None, save_path=None, show_chart=True, **kwargs):
        """
        Create a plot using matplotlib by specifying columns for the axes.

        :param x_column: Column name for the x-axis.
        :param y_column: Column name for the y-axis.
        :param plot_type: Type of plot ('scatter', 'line', 'bar'), default is 'scatter'.
        :param figsize: Tuple specifying the figure size, default is (10, 6).
        :param title: Title of the plot. If None, a default title is generated.
        :param xlabel: Label for the x-axis. If None, the x_column name is used.
        :param ylabel: Label for the y-axis. If None, the y_column name is used.
        :param save_path: Path to save the plot image. If None, the plot is not saved.
        :param kwargs: Additional keyword arguments to pass to the matplotlib plotting function.
        """
        if self.data is None:
            print("No data available to plot.")
            return

        if x_column not in self.data.columns:
            print(f"Column '{x_column}' does not exist in the data.")
            return

        if y_column not in self.data.columns:
            print(f"Column '{y_column}' does not exist in the data.")
            return

        plt.figure(figsize=figsize)

        if plot_type == 'scatter':
            plt.scatter(self.data[x_column], self.data[y_column], **kwargs)
        elif plot_type == 'line':
            plt.plot(self.data[x_column], self.data[y_column], **kwargs)
        elif plot_type == 'bar':
            plt.bar(self.data[x_column], self.data[y_column], **kwargs)
        else:
            print(f"Plot type '{plot_type}' is not supported. Choose from 'scatter', 'line', or 'bar'.")
            return

        plt.xlabel(xlabel if xlabel else x_column)
        plt.ylabel(ylabel if ylabel else y_column)
        plt.title(title if title else f"{plot_type.capitalize()} Plot of {y_column} vs {x_column}")
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        if show_chart:
            plt.show()
    
    def fit_model(self, model_type, X_columns, y_column, model_name=None, test_size=0.2, task=None, random_state=42, **kwargs):
        """
        Fit a machine learning model to the data and store it within the class.

        :param model_type: Type of model to fit ('logistic_regression', 'linear_regression', 'random_forest', 'xgboost_classifier', 'xgboost_regressor').
        :param X_columns: List of column names to be used as features.
        :param y_column: Column name to be used as the target variable.
        :param model_name: Optional name to identify the model. If not provided, model_type is used.
        :param test_size: Proportion of the dataset to include in the test split.
        :param random_state: Random state for reproducibility.
        :param kwargs: Additional keyword arguments for the model constructor.
        :return: Dictionary containing training and testing metrics.
        """
        if self.data is None:
            raise ValueError("No data available to fit the model.")

        # Check if columns exist
        for col in X_columns + [y_column]:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' does not exist in the data.")

        X = self.data[X_columns]
        y = self.data[y_column]

        # Determine if it's a classification or regression task
        if (y.dtype == 'object' or y.nunique() <= 10) and (task is None):
            task = 'classification'
        else:
            if (task is None):
                task = 'regression'

        # Split the data
        if (test_size > 0):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)
        else:
            X_train, X_test, y_train, y_test = (X, X, y, y)

        # Instantiate the model
        if model_type == 'logistic_regression' and task == 'classification':
            model = LogisticRegression(**kwargs)
        elif model_type == 'logistic_regression' and task == 'regression':
            model = LogisticRegression(**kwargs)
        elif model_type == 'linear_regression' and task == 'regression':
            model = LinearRegression(**kwargs)
        elif model_type == 'random_forest' and task == 'classification':
            model = RandomForestClassifier(**kwargs)
        elif model_type == 'random_forest' and task == 'regression':
            model = RandomForestRegressor(**kwargs)
        elif model_type == 'xgboost_classifier' and task == 'classification':
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)
        elif model_type == 'xgboost_regressor' and task == 'regression':
            model = XGBRegressor(**kwargs)
        elif model_type == 'lle' and task == 'classification':
            model = LLEClassifier(**kwargs)
        else:
            raise ValueError(f"Unsupported model_type '{model_type}' for task '{task}'.")

        # Ensure compatibility for lle (convert to numpy if needed)
        if model_type == 'lle':
            X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
            X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            y_test = y_test.values if isinstance(y_test, pd.Series) else y_test

        # Fit the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)


        # Calculate metrics
        if task == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            metrics = {'accuracy': accuracy}
            if (test_size > 0):
                print(f"Model '{model_type}' trained. Accuracy on test set: {accuracy:.4f}")
            else:
                print(f"Model '{model_type}' trained. Accuracy on training set: {accuracy:.4f}")

        else:
            mse = mean_squared_error(y_test, y_pred)
            metrics = {'mean_squared_error': mse}
            if (test_size > 0):
                print(f"Model '{model_type}' trained. Mean Squared Error on test set: {mse:.4f}")
            else:
                print(f"Model '{model_type}' trained. Mean Squared Error on training set: {mse:.4f}")


        # Store the model
        if not model_name:
            model_name = model_type
        if model_name in self.models:
            print(f"Overwriting existing model with name '{model_name}'.")
        self.models[model_name] = {
            'model': model,
            'type': model_type,
            'task': task,
            'features': X_columns,
            'target': y_column,
            'metrics': metrics
        }

        return metrics
    
    def get_model(self, model_name):
        """
        Retrieve a trained model by its name.

        :param model_name: The name of the model to retrieve.
        :return: The trained model object.
        """
        if model_name not in self.models:
            raise ValueError(f"No model found with name '{model_name}'.")
        return self.models[model_name]['model']
    
    def predict(self, model_name=None, X=None):
        """
        Make predictions using a trained model.

        :param model_name: (Optional) The name of the model to use for prediction. If not provided, the first available model is used.
        :param X: Input features as a DataFrame or array-like.
        :return: Predictions made by the model, or None if no models are available.
        """
        if X is None:
            raise ValueError("Input features 'X' must be provided for prediction.")

        if not self.models:
            print("Error: No models available for prediction.")
            return None

        if model_name is None:
            # Select the first model in the dictionary
            model_name = next(iter(self.models))
            if model_name is None:
                raise ValueError("There are no models available for prediction.")
            print(f"No model name provided. Using the first available model: '{model_name}'.")

        model = self.get_model(model_name)
        predictions = model.predict(X)
        return predictions
    
    def list_models(self):
        """
        List all trained models.

        :return: List of model names.
        """
        return list(self.models.keys())
        
    def get_all_metrics(self):
        """
        Retrieve metrics for all trained models.

        :return: Dictionary containing metrics for each model.
        """
        all_metrics = {}
        for name, details in self.models.items():
            all_metrics[name] = details['metrics']
        return all_metrics
