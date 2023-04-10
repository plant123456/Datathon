import pandas as pd
import statsmodels.formula.api as smf

# Initialises the dataframe for crashes (flag variables file)
crashes = pd.read_csv('dataset/Crashes/crash_info_flag_variables.csv',
                      usecols=['CRN', 'FATAL_OR_SUSP_SERIOUS_INJ', 'HIT_TREE_SHRUB',
                               'CORE_NETWORK', 'HIT_GDRAIL_END', 'HIT_POLE', 'ILLUMINATION_DARK'])
crashes.dropna(inplace=True)
crashes.reset_index(drop=True)

tree_crashes = crashes.query('HIT_TREE_SHRUB == 1').reset_index(drop=True)

model = smf.logit(formula='FATAL_OR_SUSP_SERIOUS_INJ ~ CORE_NETWORK + HIT_GDRAIL_END + HIT_POLE + ILLUMINATION_DARK', data=tree_crashes).fit()

print(model.summary())