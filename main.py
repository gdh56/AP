import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def fetch_data(file_dir, sheet="Historical distribution"):
    return pd.read_excel(file_dir, sheet_name=sheet)


def clean_historical_data(data, min_year=1995, year_col="Year(s) Under Review**"):
    data = data.dropna(subset=[year_col], how='all')
    # Replace years with something actually usable as years
    data[year_col] = data[year_col].apply(lambda x: int(str(x).split(",")[-1].split(" ")[-1].strip()))
    return data[data[year_col] >= min_year]


def create_historical_bar_plot(data, x_col, y_cols, year_mod=5, bar_width=4, dest="2a.png"):
    data = data[data[x_col] % year_mod == 0]
    # This just sets up an automap for colors
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(y_cols)))
    legend_list = []
    bottoms = np.linspace(0, 0, len(data[y_cols]))
    # Avoid copying and pasting the lines for each new plot
    for index, y_col in enumerate(y_cols):
        plt.bar(data[x_col], data[y_col], bottom=bottoms, color=colors[index], width=bar_width)
        if len(bottoms) == 0:
            bottoms = data[y_col]
        else:
            bottoms = bottoms + data[y_col]
    plt.xlabel(x_col)
    plt.title("Democracy Status Over Time")
    plt.legend(y_cols, loc=(0.01, 0.01))
    plt.ylabel(",".join(legend_list))
    plt.savefig(dest)
    plt.clf()


def do_historical_graph(file_dir, year_col_name, indicator_columns):
    freedom_data = fetch_data(file_dir)
    freedom_data = clean_historical_data(freedom_data)
    create_historical_bar_plot(freedom_data, year_col_name, indicator_columns)


def do_country_data(file_dir, sheet_name, ):
    data = fetch_data(file_dir, sheet=sheet_name)
    data = clean_for_countries(data)
    plot_advancing_retreating(data)


def clean_for_countries(data):
    # vicious hacks to deal with the questionable data format
    combined_dict = {"PR": dict(), "CL": dict()}
    year = 0
    for col, col_data in data.iteritems():
        col_header = col_data.at[0]
        if col_header == "Year(s) Under Review":
            continue
        if type(col_header) is str or not np.isnan(col_header):
            year = int(str(col_header).split("-")[0].split(".")[-1])
        if year == 0 or col_data[1] == "Status":
            continue
        data_type = col_data[1].strip()
        combined_dict[data_type][year] = data[col][2:]
    unit_avg_dict = dict()
    nan_var = np.NaN
    for year in combined_dict['PR'].keys():
        # Turn '-' into nan and toss out values for South Africa do to different scores for black and white people
        pr_series = combined_dict['PR'][year].apply(lambda x: nan_var if x == '-' or len(str(x)) > 1 else float(x))
        cl_series = combined_dict['CL'][year].apply(lambda x: nan_var if x == '-' or len(str(x)) > 1 else float(x))
        # 2 for the average, 7 for the max of the scale, subtracted from 1 because we want more to be better
        unit_avg_dict[year] = 1.0 - ((pr_series + cl_series) / (2.0 * 7.0))
    unit_avg_df = pd.DataFrame(unit_avg_dict)
    col_array = unit_avg_df.columns
    diffs_dict = dict()
    for i in range(1, len(col_array)):
        diffs_dict[col_array[i]] = unit_avg_df[col_array[i - 1]] - unit_avg_df[col_array[i]]
    diffs_df = pd.DataFrame(diffs_dict)
    years_advancing_receding = dict()
    for yr in diffs_df.columns:
        # get year data without nans
        year_data_wo_nans = diffs_df[yr][diffs_df[yr].notnull()]
        advancing_proportion = year_data_wo_nans[year_data_wo_nans > 0].count() / float(len(year_data_wo_nans))
        retreating_proportion = year_data_wo_nans[year_data_wo_nans < 0].count() / float(len(year_data_wo_nans))
        years_advancing_receding[yr] = {'advancing': advancing_proportion, 'retreating': retreating_proportion}
    advancing_retreating_df = pd.DataFrame(years_advancing_receding)
    return advancing_retreating_df


def plot_advancing_retreating(data):
    plt.plot(data.columns, data.T['advancing'], label='Advancing')
    plt.plot(data.columns, data.T['retreating'], label='Retreating')
    plt.ylabel("Proportion of Change")
    plt.title("Comparison of Advancing and Declining Democracies")
    plt.xlabel("Years")
    plt.legend()
    plt.savefig('2b.png')
    plt.clf()


def get_un_country_data(data_dir):
    return pd.read_csv(data_dir, error_bad_lines=False)


# Combine methods for 2c
def do_regional_data(file_dir, sheet, un_data_dir):
    data = fetch_data(file_dir, sheet=sheet)
    country_data = get_un_country_data(un_data_dir)
    proportions = clean_for_regions(data, country_data)
    plot_regional_bar(proportions)


# Combine methods for 2d
def do_develop_data(file_dir, sheet, un_data_dir):
    data = fetch_data(file_dir, sheet=sheet)
    country_data = get_un_country_data(un_data_dir)
    proportions = clean_for_develop(data, country_data)
    plot_develop_line(proportions)


# Take the data from files and put it into a format for plotting
def clean_for_regions(data, un_data):
    pre_data_frame = dict()
    pre_data_frame['Country or Area'] = data["Survey Edition"][2:]
    year = 0
    for col, col_data in data.iteritems():
        col_header = col_data.at[0]
        if col_header == "Year(s) Under Review":
            continue
        if type(col_header) is str or not np.isnan(col_header):
            year = int(str(col_header).split("-")[0].split(".")[-1])
        if year == 0 or col_data[1] != "Status":
            continue
        if year in [2005, 2020]:
            pre_data_frame[year] = data[col][2:]
    status_df = pd.DataFrame(pre_data_frame)
    combined_df = status_df.merge(un_data, how="inner", on="Country or Area")
    regions = combined_df["Region Name"].unique()
    proportions = dict()
    for region in regions:
        proportions[region] = get_democracy_proportions(combined_df[combined_df["Region Name"] == region])
    return proportions


def clean_for_develop(data, un_data):
    pre_data_frame = dict()
    pre_data_frame['Country or Area'] = data["Survey Edition"][2:]
    year = 0
    for col, col_data in data.iteritems():
        col_header = col_data.at[0]
        # This is the year column
        if col_header == "Year(s) Under Review":
            continue
        if type(col_header) is str or not np.isnan(col_header):
            # Resilient way of pulling years based on the column title that handles
            # the year spans.
            year = int(str(col_header).split("-")[0].split(".")[-1])
        # Ignore years that are before 1995 and we don't care about the status column
        if year < 1995 or col_data[1] == "Status":
            continue
        # Add a dict to hold the data for each year
        if not year in pre_data_frame.keys():
            pre_data_frame[year] = dict()
        # Create a dict of year -> PL or CR -> the column data
        pre_data_frame[year][str(col_data[1]).strip()] = col_data[2:]
    fiw_dict = {"Country or Area": pre_data_frame["Country or Area"]}
    # Build up the dictionary to convert to a dataframe
    for dict_year, metrics in pre_data_frame.items():
        if dict_year == "Country or Area":
            continue
        # Get the two types of metrics without placeholder values
        cl_series = metrics["CL"].where(metrics['CL'] != '-').dropna()
        pr_series = metrics["PR"].where(metrics['CL'] != '-').dropna()
        fiw_dict[dict_year] = (cl_series + pr_series) / 2.0
    status_df = pd.DataFrame(fiw_dict)
    combined_df = status_df.merge(un_data, how="inner", on="Country or Area")
    result = {"ldc": [], "non-ldc": []}
    # For each year get the average of the value for LDC and non-LDC countries
    for yr in range(1995, 2021):
        result['ldc'].append(np.mean(combined_df[combined_df["Least Developed Countries (LDC)"] == 'x'][yr]))
        result['non-ldc'].append(np.mean(combined_df[combined_df["Least Developed Countries (LDC)"] != 'x'][yr]))
    return result


# Helper function to iterate over the years in question and proportions
# Returns the data in dictionary form
def get_democracy_proportions(df):
    years = [2005, 2020]
    dem_levels = ["PF", "F", "NF"]
    result = dict()
    for year in years:
        result[year] = dict()
        for dem_level in dem_levels:
            result[year][dem_level] = df[df[year] == dem_level][year].count() / len(df[year][df[year] != '-'])
    return result


def plot_regional_bar(proportions):
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, 3))
    bar_width = .5
    dem_levels = ["F", "PF", "NF"]
    # Take the dictionary you made and use the bottoms aggregator to create the stacked
    # graph so it looks like parts of a whole
    for region, years in proportions.items():
        for year, dem_levels_items in years.items():
            bottoms = np.linspace(0, 0, len(proportions))
            count = -1
            for dem_level in dem_levels:
                count += 1
                # This string format keeps things alphabetical
                plt.bar("%s_%s" % (year, region), dem_levels_items[dem_level], bottom=bottoms, color=colors[count], width=bar_width)
                if len(bottoms) == 0:
                    bottoms = dem_levels_items[dem_level]
                else:
                    bottoms = bottoms + dem_levels_items[dem_level]
    plt.xlabel("Region and Year")
    plt.xticks(rotation="vertical")
    plt.title("Democracy Status Over Time By Region")
    plt.legend(dem_levels)
    plt.tight_layout()
    plt.ylabel("Proportion")
    plt.savefig("2c.png")
    plt.clf()


def plot_develop_line(averages):
    year_range = range(1995, 2021)
    plt.plot(year_range, averages['ldc'],  color='red')
    plt.plot(year_range, averages['non-ldc'], color='green')
    plt.ylabel("Average FiW Score")
    plt.title("LDC vs Non LDC FiW scores 1995-2020")
    plt.xlabel("Years")
    plt.ylabel("FiW Score")
    plt.legend(['LDC', 'Non-LDC'])
    plt.savefig('2d.png')
    plt.clf()


# Define the country data meta data
data_file_dir = "./Country_and_Territory_Ratings_and_Statuses_FIW1973-2021.xlsx"
hist_year_col_name = "Year(s) Under Review**"
hist_indicator_columns = ["% of F Countries", "% of PF Countries", "% of NF Countries"]
country_sheet = "Country Ratings, Statuses "
un_data_file = "UNSD — Methodology.csv"

if __name__ == '__main__':
    # 2a
    do_historical_graph(data_file_dir, hist_year_col_name, hist_indicator_columns)
    # 2b
    do_country_data(data_file_dir, country_sheet)
    # 2c
    do_regional_data(data_file_dir, country_sheet, un_data_file)
    # 2d
    do_develop_data(data_file_dir, country_sheet, un_data_file)
