def data_preprocessing(data):
    # Preprocess data
    # Convert time strings to datetime objects and categorize by day/night
    processed_data = {}
    n = 0
    for time in data['referenceTime']:
        processed_data[n] = {
            'referenceTime': time,
            'Period': 'Day' if 6 <= time.hour < 18 else 'Night',
            'Date': time.date()
        }
        n += 1

    data['referenceTime'] = processed_data

    # Initialize dictionaries to hold summed values and counts
    day_dict = defaultdict(lambda: {'load_NO1': 0, 'windon_NO1': 0, 'solar_NO1': 0, 'count': 0})
    night_dict = defaultdict(lambda: {'load_NO1': 0, 'windon_NO1': 0, 'solar_NO1': 0, 'count': 0})

    # Sum up the values for day and night periods
    n = 0
    for entry in data:
        if entry['referenceTime'][n]['Period'] == 'Day':
            day_dict[entry['Date']]['load_NO1'] += entry['load_NO1']
            day_dict[entry['Date']]['windon_NO1'] += entry['windon_NO1']
            day_dict[entry['Date']]['solar_NO1'] += entry['solar_NO1']
            day_dict[entry['Date']]['count'] += 1
            n += 1
        else:
            night_dict[entry['Date']]['load_NO1'] += entry['load_NO1']
            night_dict[entry['Date']]['windon_NO1'] += entry['windon_NO1']
            night_dict[entry['Date']]['solar_NO1'] += entry['solar_NO1']
            night_dict[entry['Date']]['count'] += 1
            n += 1

    # Calculate averages for each day
    final_data = {}
    for date in day_dict.keys():
        final_data[date] = {
            'load_NO1_Day': day_dict[date]['load_NO1'] / day_dict[date]['count'],
            'windon_NO1_Day': day_dict[date]['windon_NO1'] / day_dict[date]['count'],
            'solar_NO1_Day': day_dict[date]['solar_NO1'] / day_dict[date]['count'],
            'load_NO1_Night': night_dict[date]['load_NO1'] / night_dict[date]['count'],
            'windon_NO1_Night': night_dict[date]['windon_NO1'] / night_dict[date]['count'],
            'solar_NO1_Night': night_dict[date]['solar_NO1'] / night_dict[date]['count'],
        }

    # Convert the final_data dictionary to a DataFrame for easy viewing
    final_df = pd.DataFrame.from_dict(final_data, orient='index').reset_index().rename(columns={'index': 'Date'})

    return final_df