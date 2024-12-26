from scipy.stats import kruskal
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
import numpy as np
import matplotlib.pyplot as plt

def obtener_win(sig, binary_sig, siPlot=True):
        # Ensure binary_sig is binary
        binary_sig = np.array(binary_sig).flatten()
        binary_sig = (binary_sig >= 0.5).astype(int)  # Convert to binary (0 or 1)

        diff = np.diff(binary_sig)
        idx_actividad = np.where(diff == 1)[0]
        idx_rep = np.where(diff == -1)[0]

        print(binary_sig.shape)

        if binary_sig[0] == 1:
            idx_actividad = np.insert(idx_actividad, 0, 0)
        #if binary_sig[-1] == 1:
        #    idx_actividad = np.append(idx_actividad, len(binary_sig) -1)
        
        if binary_sig[0] == 0:
            idx_rep = np.insert(idx_rep, 0, 0)
        #if binary_sig[-1] == 0:
        #    idx_rep = np.append(idx_rep, len(binary_sig) - 1)
        
        #print(idx_actividad)
        #print(idx_rep)
        
        # Ensure idx_actividad and idx_rep have the same length by adding samples
        while len(idx_actividad) < len(idx_rep):
            idx_actividad = np.append(idx_actividad, idx_actividad[-1])
        while len(idx_rep) < len(idx_actividad):
            idx_rep = np.append(idx_rep, idx_rep[-1])
        
        if idx_rep[0] < idx_actividad[0]:
            ventanas_reposo = np.stack((idx_rep, idx_actividad)).T
            ventanas_actividad = np.stack((idx_actividad[:-1], idx_rep[1:])).T
        else:
            ventanas_reposo = np.stack((idx_rep[:-1], idx_actividad[1:])).T
            ventanas_actividad = np.stack((idx_actividad[:], idx_rep[:])).T

        #print(ventanas_reposo.shape)
        #print(ventanas_actividad.shape)

        if siPlot:
            plt.figure(figsize=(20, 5))
            plt.subplot(1, 2, 1) 
            for ventana in ventanas_actividad:
                plt.plot(sig[ventana[0]: ventana[1]])
            plt.title('Ventanas de actividad')

            plt.subplot(1, 2, 2)  
            for ventana in ventanas_reposo:
                plt.plot(sig[ventana[0]: ventana[1]])
            plt.title('Ventanas de reposo')

            plt.show()

        return ventanas_actividad, ventanas_reposo


def compute_kruskal_wallis_anova(df, group_column, value_column):
    """
    Computes the Kruskal-Wallis H-test for given groups in a DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - group_column: Column name in df that contains the group labels.
    - value_column: Column name in df that contains the values to compare across groups.

    Returns:
    - The test statistic and p-value from the Kruskal-Wallis H-test.
    """
    
    groups = df[group_column].unique()
    
    group_data = [df[df[group_column] == group][value_column] for group in groups]
    
    test_stat, p_value = kruskal(*group_data)
    
    return test_stat, p_value


def compute_one_way_anova(df, group_column, value_column):
    """
    Computes the one-way ANOVA test for given groups in a DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - group_column: Column name in df that contains the group labels.
    - value_column: Column name in df that contains the values to compare across groups.

    Returns:
    - The F statistic and p-value from the one-way ANOVA test.
    """
    
    groups = df[group_column].unique()
    
    group_data = [df[df[group_column] == group][value_column] for group in groups]
    
    F_stat, p_value = f_oneway(*group_data)
    
    return F_stat, p_value


def compute_two_way_anova(df, dependent_var, factor1, factor2):
    """
    Computes the two-way ANOVA for a given DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - dependent_var: The name of the dependent variable (continuous).
    - factor1: The name of the first factor (independent variable).
    - factor2: The name of the second factor (independent variable).

    Returns:
    - ANOVA table as a DataFrame.
    """
    # Construct the formula for the two-way ANOVA
    formula = f'{dependent_var} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})'
    
    # Fit the model
    model = ols(formula, data=df).fit()
    
    # Perform ANOVA and return the table
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table
def compute_t_test(df, group_column, value_column):
    """
    Computes the t-test for the means of two independent groups in a DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - group_column: Column name in df that contains the group labels.
    - value_column: Column name in df that contains the values to compare.
    - group1_label: The label of the first group for comparison.
    - group2_label: The label of the second group for comparison.

    Returns:
    - The T statistic and p-value from the t-test.
    """
    groups = df[group_column].unique()
    if(groups.shape[0] != 2):
        raise ValueError("Los gurpos deben de ser unicamente dos")
    # Extract data for the two specified groups
    group1_data = df[df[group_column] == groups[0]][value_column]
    group2_data = df[df[group_column] == groups[1]][value_column]
    
    # Perform the t-test
    T_stat, p_value = ttest_ind(group1_data, group2_data)
    
    return T_stat, p_value



def compute_paired_t_test(df, group_column, value_column):
    """
    Computes the paired t-test for the means of two related groups in a DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - group_column: Column name in df that contains the group labels.
    - value_column: Column name in df that contains the values to compare.

    Returns:
    - The T statistic and p-value from the paired t-test.
    """
    groups = df[group_column].unique()
    if len(groups) != 2:
        raise ValueError("Los gurpos deben de ser unicamente dos")
    
    # Extract data for the two specified groups
    group1_data = df[df[group_column] == groups[0]][value_column]
    group2_data = df[df[group_column] == groups[1]][value_column]
    
    # Ensure both groups have the same number of observations
    if len(group1_data) != len(group2_data):
        raise ValueError("Both groups must have the same number of observations for a paired t-test.")
    
    # Perform the paired t-test
    T_stat, p_value = ttest_rel(group1_data, group2_data)
    
    return T_stat, p_value