import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime

def plot_data(df):
    """
    Plotting data as one common plot with a subplot (bar plot or histogram) for each column\n
    Column 'age' is in days, changed to years for visualization\n
    Calculating the number of bins for each histogram through Freedman–Diaconis rule\n
    Parameters:\n
    - df (DataFrame): Pandas DataFrame to be plotted/visualized\n
    """
    bar_plots = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
    histo_plots = ['age_years', 'age', 'bmi', 'height', 'weight', 'ap_hi', 'ap_lo']
    columns_to_keep = set(bar_plots).union(histo_plots)
    df_cl = df.drop(columns=[col for col in df.columns if col not in columns_to_keep])
    # change column 'age' from days to years for visualization
    if 'age' in df_cl:
        df_cl.insert(1, 'age_years', df['age'] // 365)
        df_cl.drop(columns=['age'], inplace=True)

    plt.rc('font', size=6)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#287b8f'])
    
    common_plot = plt.figure(figsize=(10,10))
    number_cols = 4 if len(df_cl.columns) > 9 else 3
    i = 1
    for column_name in df_cl:
        current_plot = common_plot.add_subplot(math.ceil(len(df_cl.columns)/number_cols), number_cols, i)
        if column_name in bar_plots:
            val_counts = df_cl[column_name].value_counts()
            current_plot.bar(val_counts.index, val_counts.values)
            current_plot.set_title('Bar Plot of ' + column_name.capitalize())
            current_plot.set_xlabel(column_name.capitalize())
            current_plot.set_xticks(val_counts.index)
            current_plot.set_xticklabels(val_counts.index)
            current_plot.set_ylabel('Count')
        if column_name in histo_plots: 
            q25, q75 = np.percentile(df_cl[column_name], [25, 75])
            bin_width = 2 * (q75 - q25) * len(df_cl[column_name]) ** (-1/3)
            bin_width = max(bin_width, 1)
            bins = int((df_cl[column_name].max() - df_cl[column_name].min()) / bin_width)
            # for outlier (blood pressure)
            bins = bins if bins < 100 else 100
            current_plot.hist(df_cl[column_name], bins=bins)
            current_plot.set_title('Histogram of ' + column_name.capitalize())
            current_plot.set_xlabel(column_name.capitalize())
            current_plot.set_ylabel('Frequency')
        i += 1
    common_plot.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.show()
    plt.rc('font', size=10)

def clean_data(df):
    """
    Cleaning data by removing records with unrealistic values for height, weight or blood pressure\n
    Parameters:\n
    - df (DataFrame): Pandas DataFrame to be cleaned\n
    """
    print('Dimensions before data cleaning: rows: ' + str(df.shape[0]) + ' cols: ' + str(df.shape[1]))
    df.query('height > 140 & height < 210', inplace=True)
    df.query('weight > 45 & weight < 190', inplace=True)
    df.query('ap_hi > 80 & ap_hi < 200', inplace=True)
    df.query('ap_lo > 40 & ap_lo < 150', inplace=True)
    df.query('ap_hi > ap_lo', inplace=True)
    print('Dimensions after data cleaning: rows: ' + str(df.shape[0]) + ' cols: ' + str(df.shape[1]))

def optimize_data(df):
    """
    Optimizing data by adding BMI column instead of weight and height, and dropping columns which 
    were not relevant to the model\n
    Parameters:\n
    - df (DataFrame): Pandas DataFrame to be optimized\n
    """
    df.insert(3, 'bmi', df['weight'] / (df['height'] / 100) ** 2)
    df.drop(columns=['id', 'weight', 'height', 'smoke', 'alco'], inplace=True)

def plot_correlation(df):
    """
    Plotting a correlation matrix for all columns of the DataFrame, including each correlation coefficient\n
    Parameters:\n
    - df (DataFrame): Pandas DataFrame containing the data that should be plotted\n
    """
    corr = df.corr()
    plt.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, '{:.2f}'.format(corr.iloc[i, j]), ha='center', va='center', color='white')
    plt.colorbar()
    plt.show()

def train_model(x_train, y_train):
    """
    Building the Logit model and fitting (training) the data\n
    Parameters:\n
    - x_train (DataFrame): Pandas DataFrame containing the independent (regressor) variables for training set\n
    - y_train (DataFrame): Pandas DataFrame containing the binary target variable for training set\n
    Returns\n
    - log_reg (BinaryResultsWrapper): fitted logistic regression model\n
    """
    logit = sm.Logit(y_train, x_train)
    log_reg = logit.fit()
    return log_reg

def plot_confusion_accuracy(y_test, prediction):
    """
    Plotting the Confusion Matrix and printing the model accuracy\n
    Parameters:\n
    - y_test (DataFrame): Pandas DataFrame containing the binary target variable for testing set, containig 
    true value\n
    - prediction (list): containing predicted value for records of y_test\n
    """
    cm = confusion_matrix(y_test, prediction)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()
    print('Test accuracy = ', accuracy_score(y_test, prediction))

def test_model(log_reg, x_test, y_test):
    """
    Testing the fitted model and calling function to plot confusion matrix and print accuracy\n
    Parameters:\n
    - log_reg (BinaryResultsWrapper): fitted logistic regression model\n
    - x_test (DataFrame): Pandas DataFrame containing the independent (regressor) variables for testing set\n
    - y_test (DataFrame): Pandas DataFrame containing the binary target variable for testing set\n
    """
    yhat = log_reg.predict(x_test) 
    prediction = list(map(round, yhat))
    plot_confusion_accuracy(y_test, prediction)

def print_model_function(log_reg):
    """
    Printing the probability function of the fitted model\n
    Parameters:\n
    - log_reg (BinaryResultsWrapper): fitted logistic regression model\n
    """
    coefficients = log_reg.summary2().tables[1]['Coef.']
    template = "({}*age) + ({}*gender) + ({}*bmi) + ({}*ap_hi) + ({}*ap_lo) + ({}*cholesterol) + ({}*gluc) + ({}*active)"
    z_string = template.format("{:.4f}".format(coefficients['age']), "{:.4f}".format(coefficients['gender']), "{:.4f}".format(coefficients['bmi']), "{:.4f}".format(coefficients['ap_hi']), "{:.4f}".format(coefficients['ap_lo']), "{:.4f}".format(coefficients['cholesterol']), "{:.4f}".format(coefficients['gluc']), "{:.4f}".format(coefficients['active']))
    print("\nz = " + z_string)
    p_string = "P = 1 / (1+e^(-{}))".format(z_string)
    print("P = 1 / (1+e^(-z))")
    print("Model function:")
    print(p_string + "\n")

def print_hi_lo_entries(log_reg, df, n=5):
    """
    Printing top 5 entries with the highest as well as lowest risk of cardio\n
    Parameters:\n
    - log_reg (BinaryResultsWrapper): fitted logistic regression model\n
    - df (DataFrame): Pandas DataFrame containing the records\n
    """
    df['prediction'] = log_reg.predict(df.drop(columns=['cardio']))
    df_sorted = df.sort_values(by='prediction', ascending=False)
    print("Top 5 entries with the highest risk of cardio:")
    print(df_sorted.head(n))
    print("Top 5 entries with the lowest risk of cardio:")
    print(df_sorted.tail(n))

def predict_df(log_reg, df):
    df['prediction'] = log_reg.predict(df)
    print(df)

def start_predictions(log_reg):
    while True:
        user_input = input("Want to predict data? (type 'stop' to end, 'start' to enter data): ")
        if user_input.lower() == 'stop':
            break
        if user_input.lower() == 'start':
            birthday_input = input("Please enter your date of birth (dd.mm.yyyy): ")
            birthday = datetime.strptime(birthday_input, "%d.%m.%Y")
            current_date = datetime.now()
            age = int((current_date - birthday).days)
            gender = int(input("Please enter your gender (1 for female, 2 for male): "))
            weight = float(input("Please enter your weight (kg): "))
            height = float(input("Please enter your height (cm): "))
            bmi = weight / (height / 100) ** 2        
            ap_hi = int(input("Please enter your systolic blood pressure: "))
            ap_lo = int(input("Please enter your diastolic blood pressure: "))
            cholesterol = int(input("Please enter your cholesterol level (1 for normal, 2 for above normal, 3 for well above normal): "))
            gluc = int(input("Please enter your glucose level (1 for normal, 2 for above normal, 3 for well above normal): "))
            active = int(input("Please enter whether or not you are physically active (0 for no, 1 for yes): "))

            columns = ['age', 'gender', 'bmi', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'active']
            data = [[age, gender, bmi, ap_hi, ap_lo, cholesterol, gluc, active]]
            df_input = pd.DataFrame(data, columns=columns)
            predict_df(log_reg, df_input)

df = pd.read_csv('cardio.csv', delimiter=';') # (70.000, 13)
plot_data(df) 
clean_data(df) # (67.850, 13)
optimize_data(df) # (67.850, 9)
plot_data(df)
plot_correlation(df)

# split data in training and test set (70:30)
x = df.drop(columns=['cardio'])
y = df['cardio']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

log_reg = train_model(x_train, y_train)
test_model(log_reg, x_test, y_test)

print_model_function(log_reg)
print_hi_lo_entries(log_reg, df, 5)

start_predictions(log_reg)