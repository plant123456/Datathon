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
            'HIT_GDRAIL', 'HIT_GDRAIL_END', 'HIT_POLE', 'HIT_TREE_SHRUB', 'ICY_ROAD',
            'ILLUMINATION_DARK', 'INTERSECTION', 'INTERSTATE', 'LIMIT_65MPH',
            'LOCAL_ROAD', 'LOCAL_ROAD_ONLY', 'OTHER_FREEWAY', 'RAMP', 'SHLDR_RELATED',
            'SIGNALIZED_INT', 'SNOW_SLUSH_ROAD', 'STOP_CONTROLLED_INT',
            'UNSIGNALIZED_INT', 'WET_ROAD', 'WORK_ZONE']

# Counts the number of fatal crashes
fatal_crashes = crashes.query('FATAL_OR_SUSP_SERIOUS_INJ == 1').reset_index(drop=True)
number_fatal = len(fatal_crashes)

# Counts fatality ratio for all possible reasons related to roads
for reason in variables:

    # Df for all crashes
    crash_reason = crashes.query(f'{reason} == 1').reset_index(drop=True)
    number_crash_reason = len(crash_reason)
    print(f"Crashes due to {reason}:", number_crash_reason)
    print()

    # Df for all crashes which are fatal
    fatal_crash_reason = fatal_crashes.query(f'{reason} == 1').reset_index(drop=True)
    number_fatal_crash_reason = len(fatal_crash_reason)
    print(f"Fatal crashes due to {reason}:", number_fatal_crash_reason)

    print("Ratio of fatality:", (number_fatal_crash_reason / number_crash_reason))
    print()