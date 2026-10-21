# Logistics Data Analytics Project

## Project Overview

This project focuses on analyzing logistics data using Python. It covers four weeks of tasks, starting with basic delivery performance analysis and progressing to data preprocessing, advanced visualization, and predictive modeling.

The main goal is to understand logistics performance, identify delivery delays, explore important patterns, and use machine learning to support better operational decisions.

All datasets used in this project are sample or simulated data created for learning purposes.

## Tools and Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Visual Studio Code
- GitHub

## Week 1: Logistics Strategic Planning

### Objective
Analyze sample delivery data to understand delivery performance and identify delayed orders.

### Analysis Performed
- Displayed the delivery dataset.
- Calculated delivery delays.
- Identified delayed orders.
- Calculated average actual delivery time.
- Found the order with the highest delay.
- Created a bar chart of actual delivery times.

### Key Findings
- Average actual delivery time was approximately 4.33 hours.
- Orders 102, 104, and 105 were delayed.
- Order 104 had the highest delay of 2 hours.

### Conclusion
Python helped analyze delivery performance and identify delays that may require better planning.

## Week 2: Data Collection, Cleaning, and Preprocessing

### Objective
Prepare logistics data for analysis by checking data quality and applying data cleaning and preprocessing techniques.

### Activities
- Collected or prepared sample logistics data.
- Inspected the dataset and its structure.
- Checked for missing values and duplicate records.
- Examined data quality issues.
- Applied appropriate cleaning and preprocessing steps.
- Generated output and a histogram using Python.

### Conclusion
Data cleaning and preprocessing help improve data quality and make datasets more suitable for reliable analysis.

## Week 3: Advanced Data Analysis and Visualization in Logistics

### Objective
Explore logistics data using descriptive statistics, group analysis, correlations, and visualizations.

### Analysis Performed
- Generated a hypothetical logistics dataset.
- Calculated mean, median, and standard deviation.
- Checked missing values and duplicate records.
- Compared average transportation costs by transport mode.
- Analyzed delivery times across regions.
- Examined correlations between numerical logistics variables.
- Created six visualizations.

### Visualizations
1. Distribution of delivery times.
2. Average transportation cost by transport mode.
3. On-time versus delayed shipments.
4. Distance versus delivery time.
5. Correlation heatmap.
6. Average delivery time by region.

### Conclusion
Visualizations help identify patterns, compare logistics performance, understand relationships between variables, and recognize areas that may need operational improvement.

## Week 4: Predictive Modeling and Optimization in Logistics Systems

### Objective
Build machine learning models to predict delivery time and explore ways to improve logistics planning.

### Dataset Features
- Distance in kilometres.
- Shipment weight.
- Transport mode.
- Traffic level.
- Weather conditions.
- Delivery time as the prediction target.

### Models and Techniques
- Linear Regression.
- Random Forest Regression.
- Training and testing data split.
- Five-fold cross-validation for model tuning.
- GridSearchCV for hyperparameter selection.

### Model Evaluation
The models are evaluated using:

- Mean Absolute Error (MAE).
- Root Mean Squared Error (RMSE).
- R-squared (R²).

The model comparison results are saved for review.

### Visualizations and Outputs
- Actual versus predicted delivery times.
- Model comparison using MAE.
- Distribution of prediction errors.
- Random Forest feature importance.
- Predicted shipment times and schedule-review flags.

### Optimization Recommendations
- Review shipments predicted to require more delivery time.
- Use delivery-time predictions to support scheduling decisions.
- Consider traffic, distance, weather, and transport mode during planning.
- Use model performance results to evaluate the usefulness of predictions.

### Conclusion
Predictive modeling can support delivery planning and help logistics teams identify shipments that may require additional attention. The results from this project are based on simulated data and would need validation using real logistics records before operational use.

## Overall Conclusion

This four-week project demonstrates how Python can be used throughout a logistics data analytics workflow, from basic analysis and data preprocessing to visualization and predictive modeling.

The project helped develop practical skills in data cleaning, exploratory data analysis, visualization, machine learning, model evaluation, and data-driven decision-making.

## How to Run the Project

1. Install Python on your computer.
2. Install the required libraries:

   `python -m pip install pandas numpy matplotlib scikit-learn`

3. Open the required Python script in Visual Studio Code.
4. Run the script using the VS Code terminal.
5. Review the generated charts, datasets, and model evaluation results.

## Note

The datasets in this project are hypothetical and intended for educational purposes. The findings illustrate analytical techniques and should not be treated as verified results from a real logistics company.
