from scipy.stats import kruskal
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
import numpy as np

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