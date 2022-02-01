import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def fetch_data(file_dir, sheet="Historical distribution"):
    return pd.read_excel(file_dir, sheet_name=sheet)


def clean_historical_data(data, min_year=1995, year_col="Year(s) Under Review**"):
    data = data.dropna(subset=[year_col], how='all')
    data[year_col] = data[year_col].apply(lambda x: int(str(x).split(",")[-1].split(" ")[-1].strip()))
    return data[data[year_col] >= min_year]


def create_historical_bar_plot(data, x_col, y_cols, year_mod=5, bar_width=4, dest="historical.png"):
    data = data[data[x_col] % year_mod == 0]
    cmap = plt.get_cmap("Reds")
    colors = cmap(np.linspace(0, 1, len(y_cols)))
    legend_list = []
    bottoms = np.linspace(0, 0, len(data[y_cols]))
    for index, y_col in enumerate(y_cols):
        plt.bar(data[x_col], data[y_col], bottom=bottoms, color=colors[index], width=bar_width)
        if len(bottoms) == 0:
            bottoms = data[y_col]
        else:
            bottoms = bottoms + data[y_col]
    plt.xlabel(x_col)
    plt.legend(y_cols, loc=(0.01,0.01))
    plt.ylabel(",".join(legend_list))
    plt.savefig(dest)


def do_historical_graph(file_dir, year_col_name, indicator_columns):
    freedom_data = fetch_data(file_dir)
    freedom_data = clean_historical_data(freedom_data)
    create_historical_bar_plot(freedom_data, year_col_name, indicator_columns)


def do_country_data(file_dir, sheet_name, ):
    data = fetch_data(file_dir, sheet=sheet_name)
    clean_for_countries(data)


def clean_for_countries(data):
    # vicious hacks to deal with the questionable data format
    combined_dict = {"PR": dict(), "CL": dict()}
    year = 0
    for col, col_data in data.iteritems():
        col_header = col_data.at[0]
        if not type(col_header) is str and not np.isnan(col_header):
            year = int(str(col_header).split("-")[0].split(".")[-1])
        if year == 0 or col_data[1] == "Status":
            continue
        data_type = col_data[1].strip()
        combined_dict[data_type][year] = col_data[2:]
    print("here")




data_file_dir = "./Country_and_Territory_Ratings_and_Statuses_FIW1973-2021.xlsx"
hist_year_col_name = "Year(s) Under Review**"
hist_indicator_columns = ["% of F Countries", "% of PF Countries", "% of NF Countries"]
country_sheet = "Country Ratings, Statuses "

if __name__ == '__main__':
    #do_historical_graph(data_file_dir, hist_year_col_name, hist_indicator_columns)
    do_country_data(data_file_dir, country_sheet)
