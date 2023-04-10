import pandas as pd

# Initialises the dataframe for crashes (flag variables file)
crashes = pd.read_csv('dataset/Crashes/crash_info_flag_variables.csv',
                      usecols=['CRN', 'FATAL_OR_SUSP_SERIOUS_INJ',
                               'CORE_NETWORK', 'CURVED_ROAD', 'HIT_BARRIER', 'HIT_BRIDGE', 'HIT_EMBANKMENT',
                               'HIT_GDRAIL', 'HIT_GDRAIL_END', 'HIT_POLE', 'HIT_TREE_SHRUB', 'ICY_ROAD',
                               'ILLUMINATION_DARK', 'INTERSECTION', 'INTERSTATE', 'LIMIT_65MPH', 'LIMIT_70MPH',
                               'LOCAL_ROAD', 'LOCAL_ROAD_ONLY', 'OTHER_FREEWAY', 'RAMP', 'SHLDR_RELATED',
                               'SIGNALIZED_INT', 'SNOW_SLUSH_ROAD', 'STOP_CONTROLLED_INT', 'TURNPIKE',
                               'UNSIGNALIZED_INT', 'WET_ROAD', 'WORK_ZONE'])
crashes.dropna(inplace=True)
crashes.reset_index(drop=True)

# Initialises list of all possible variables (removed LIMIT_70MPH and TURNPIKE as no crashes related to these variables)
variables = ['CORE_NETWORK', 'CURVED_ROAD', 'HIT_BARRIER', 'HIT_BRIDGE', 'HIT_EMBANKMENT',
            'HIT_GDRAIL', 'HIT_GDRAIL_END', 'HIT_POLE', 'ICY_ROAD',
            'ILLUMINATION_DARK', 'INTERSECTION', 'INTERSTATE', 'LIMIT_65MPH',
            'LOCAL_ROAD', 'LOCAL_ROAD_ONLY', 'OTHER_FREEWAY', 'RAMP', 'SHLDR_RELATED',
            'SIGNALIZED_INT', 'SNOW_SLUSH_ROAD', 'STOP_CONTROLLED_INT',
            'UNSIGNALIZED_INT', 'WET_ROAD', 'WORK_ZONE']

# Counts the number of fatal crashes
crashes_with_trees = crashes.query('HIT_TREE_SHRUB == 1').reset_index(drop=True)
number_crashes = len(crashes_with_trees)

# Counts the number of crashes involving trees which resulted in injuries
fatal_crashes_with_trees = crashes_with_trees.query('FATAL_OR_SUSP_SERIOUS_INJ == 1').reset_index(drop=True)
number_fatal_with_trees = len(fatal_crashes_with_trees)

# Calculates the fatality rate of crashes involving trees
fatality_rate_trees = number_fatal_with_trees / number_crashes

print("Fatality rate when crashed into trees:", fatality_rate_trees)

# Calculates which variables are significant
for reason in variables:

    # With tree + with road condition
    crash_with_reason = crashes_with_trees.query(f'{reason} == 1').reset_index(drop=True)
    if len(crash_with_reason) == 0:
        continue

    fatal_crash_with_reason = fatal_crashes_with_trees.query(f'{reason} == 1').reset_index(drop=True)
    ratio = len(fatal_crash_with_reason) / len(crash_with_reason)

    if ratio > fatality_rate_trees:
        print(f"Fatality rate when {reason}:", ratio, f"Count: {len(fatal_crash_with_reason)}")