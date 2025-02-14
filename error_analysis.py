from sklearn.metrics import precision_recall_curve, auc, classification_report
import pandas as pd
import json
import numpy as np
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, Panel, Tabs
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10
from bokeh.io import output_file
from bokeh.models import TabPanel  # Import TabPanel
from bokeh.transform import dodge

class error_analysis():
    def __init__(self):
        pass
    
    def compute_final_df(self):  # Add self as the first parameter
        predictions = pd.read_csv('data/Predictions/predictions_valcatboost.csv')
        predictions_proba = pd.read_csv('data/Predictions/predictions_proba_valcatboost.csv')
        y_test = pd.read_pickle('data/processed/y_test.pkl')
        X_test = pd.read_pickle('data/processed/X_test.pkl')
        print("Unique values in y_test:", np.unique(y_test))

        # Ensure y_test is binary and predictions_proba has shape (n_samples, 2)
        if predictions_proba.shape[1] != 2:
            raise ValueError("predictions_proba must have shape (n_samples, 2).")
        if not np.all(np.isin(y_test, [0, 1])):
            raise ValueError("y_test must contain binary labels (0 or 1).")

        # Entropy calculation
        entropy = -np.sum(predictions_proba * np.log(predictions_proba + 1e-10), axis=1)  # Add small epsilon to avoid log(0)

        # Brier score calculation (for both classes)
        y_pred_proba = predictions_proba.iloc[:, 1]  # Probabilities for class 1
        brier = (y_pred_proba - y_test) ** 2
        print("Brier Score:", np.mean(brier))
        calibration_error = np.abs(predictions_proba.iloc[:, 1] - y_test)

        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions_proba.iloc[:, 1], bin_edges) - 1  # Use .iloc for column access
        ece = 0
        for bin_idx in range(n_bins):
            in_bin = bin_indices == bin_idx
            if np.any(in_bin):
                bin_conf = predictions_proba.iloc[in_bin, 1].mean()
                bin_acc = y_test[in_bin].mean()
                ece += np.abs(bin_conf - bin_acc) * in_bin.sum()
        ece /= len(y_test)

        # Add entropy, brier_score, and ece directly to X_test
        X_test['entropy'] = entropy
        X_test['brier_score'] = brier
        X_test['calibration_error'] = calibration_error

        predictions = predictions.squeeze()
        # Concatenate all data (if needed)
        df = pd.concat([X_test.reset_index(drop=True), 
                        y_test.reset_index(drop=True).rename('label'),
                        predictions.reset_index(drop=True).rename('prediction'), 
                        predictions_proba.reset_index(drop=True)], axis=1)

        # Print and return results
        print(df.head())
        print(f"ECE: {ece:.4f}")
        return df, ece
    
    def plot_error_analysis(self, df):  # Add self as the first parameter
        # Plot ECE vs. entropy
        sns.scatterplot(data=df, x='entropy', y='brier_score', hue='label')
        plt.xlabel('Entropy')
        plt.ylabel('Brier Score')
        plt.title('Brier Score vs. Entropy')
        plt.show()
    
    def create_interactive_plot(self, df, categorical_columns):
        """
        Create an interactive Bokeh plot with panels for different categorical columns.
        
        Parameters:
        - df: DataFrame containing the data.
        - categorical_columns: List of categorical columns to analyze (e.g., ['product', 'campaign_id']).
        """
        # Prepare the data for each categorical column
        panels = []
        for col in categorical_columns:
            # Group by the categorical column and calculate mean entropy and brier score
            grouped = df.groupby(col, observed=False).agg({
                'entropy': 'mean',
                'brier_score': 'mean'
            }).reset_index()

            # Create a ColumnDataSource for Bokeh
            source = ColumnDataSource(grouped)

            # Create a figure
            p = figure(
                x_range=[str(x) for x in grouped[col].tolist()],  # Use categories as x-axis
                title=f"Average Entropy and Brier Score by {col}",
                x_axis_label=col,
                y_axis_label="Value",
                width=800,
                height=400,
                tools="pan,wheel_zoom,box_zoom,reset"
            )
            entropy_color = Category10[10][1]  # Darker blue
            brier_color = Category10[10][0]  # Orange
            # Add bars for average entropy
            entropy_bar = p.vbar(
                x = dodge(col, -0.1, range=p.x_range),
                top='entropy',
                width=0.2,
                source=source,
                legend_label="Average Entropy",
                color=entropy_color,
                alpha=0.6
            )

            # Add bars for average brier score
            brier_bar = p.vbar(
                x= dodge(col, 0.1, range=p.x_range),
                top='brier_score',
                width=0.2,
                source=source,
                legend_label="Average Brier Score",
                color=brier_color,
                alpha=0.6
            )
            
            # Adjust legend font size
            p.legend.label_text_font_size = "7pt"  # Change font size

            # Adjust legend box size
            p.legend.glyph_width = 5  # Adjust width of color swatches
            p.legend.glyph_height = 5  # Adjust height of color swatches
            p.legend.label_text_font_style = "italic"
            # Optionally, change legend location
            p.legend.location = "top_right"  # You can set this to "top_left", "bottom_left", etc.


            # Add hover tool
            hover = HoverTool()
            hover.tooltips = [
                (col, f"@{col}"),
                ("Average Entropy", "@entropy{0.000}"),
                ("Average Brier Score", "@brier_score{0.000}")
            ]
            p.add_tools(hover)


            # Create a panel for this categorical column
            

            panel = TabPanel(child=p, title=col)  # Use TabPanel instead of Panel




            panels.append(panel)

        # Create tabs
        tabs = Tabs(tabs=panels)

        # Show the plot
        output_file("error_analysis.html")
        show(tabs)


if __name__ == '__main__':
    # Create an instance of error_analysis
    ea = error_analysis()
    
    # Call instance methods
    df, ece = ea.compute_final_df()
    df.to_csv('data/Predictions/error_analysis.csv', index=False)
    #ea.plot_error_analysis(df)

    categorical_columns = ['product', 'campaign_id', 'user_group_id', 'age_level', 'user_depth', 'city_development_index']

    # Create and display the interactive Bokeh plot
    ea.create_interactive_plot(df, categorical_columns)