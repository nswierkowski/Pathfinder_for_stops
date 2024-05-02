import pandas as pd

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


def create_dijkstra_diagram() -> None:
    file_path = '../dijkstra_results_2.csv'
    df = pd.read_csv(file_path)

    column1 = 'Distance'
    column2 = 'Executing time'

    X = df[[column1]]
    y = df[column2]

    plt.scatter(X, y, label='Data')
    plt.title('Scatter Plot for dijsktra algorithm')
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.legend()
    plt.grid(True)

    diagram_file = 'scatter_plot_dijkstra.png'
    plt.savefig(diagram_file)


def create_dijkstra_diagram_stops() -> None:
    file_path = '../dijkstra_results_2.csv'
    df = pd.read_csv(file_path)

    column1 = 'Stops number'
    column2 = 'Executing time'

    X = df[[column1]]
    y = df[column2]

    plt.scatter(X, y, label='Data', color='green')
    plt.title('Scatter Plot for dijsktra algorithm')
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.legend()
    plt.grid(True)

    diagram_file = 'scatter_plot_dijkstra_stops.png'
    plt.savefig(diagram_file)


def create_dijkstra_median_diagram(bucket_size: float = 0.01) -> None:
    file_path = '../dijkstra_results_2.csv'
    df = pd.read_csv(file_path)

    column1 = 'Distance'
    column2 = 'Executing time'

    # Group distances into buckets of given size and calculate median of values in each bucket
    df['Bucket'] = (df[column1] // bucket_size) * bucket_size
    median_values = df.groupby('Bucket')[column2].median()

    # Plot the data
    plt.plot(median_values.index, median_values.values, marker='o', linestyle='-')
    plt.title(f'Median Executing Time vs. Distance (Bucket Size: {bucket_size})')
    plt.xlabel('Manhattan distance')
    plt.ylabel('Median Executing Time (millis)')
    plt.grid(True)

    diagram_file = f'median_executing_time_vs_distance_bucket_{bucket_size}.png'
    plt.savefig(diagram_file)
    plt.show()


def create_dijkstra_median_diagram_stops(bucket_size: int = 3) -> None:
    file_path = '../dijkstra_results_2.csv'
    df = pd.read_csv(file_path)

    column1 = 'Stops number'
    column2 = 'Executing time'

    df['Bucket'] = (df[column1] // bucket_size) * bucket_size
    median_values = df.groupby('Bucket')[column2].median()

    plt.plot(median_values.index, median_values.values, marker='o', linestyle='-', color='green')
    plt.title(f'Median Executing Time vs. Number of stops')
    plt.xlabel('Number of stops')
    plt.ylabel('Median Executing Time (millis)')
    plt.grid(True)

    diagram_file = f'median_executing_time_vs_stops_number.png'
    plt.savefig(diagram_file)


def create_time_on_distance_buckets(bucket_size: float = 0.01) -> None:
    file_path = '../dijkstra_results_2.csv'
    df = pd.read_csv(file_path)

    column1 = 'Distance'
    column2 = 'Travel time'

    df['Bucket'] = (df[column1] // bucket_size) * bucket_size
    median_values = df.groupby('Bucket')[column2].median()

    plt.plot(median_values.index, median_values.values, marker='o', linestyle='-', color='red')
    plt.title(f'Median Travel Time vs. Distance')
    plt.xlabel('Manhattan distance between stops')
    plt.ylabel('Travel time (in seconds)')
    plt.grid(True)

    diagram_file = f'median_travel_time_vs_distance.png'
    plt.savefig(diagram_file)


def create_travel_time_vs_distance() -> None:
    file_path = '../dijkstra_results_2.csv'
    df = pd.read_csv(file_path)

    column1 = 'Distance'
    column2 = 'Travel time'

    X = df[[column1]]
    y = df[column2]

    plt.scatter(X, y, color='red')
    plt.title('Travel Time vs. Distance')
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.legend()
    plt.grid(True)

    diagram_file = 'travel_time_vs_distance.png'
    plt.savefig(diagram_file)


def create_time_on_hour_buckets(bucket_size: int = 1200) -> None:
    file_path = '../dijkstra_results.csv'
    df = pd.read_csv(file_path)

    column1 = 'Starting hour'
    column2 = 'Travel time'

    df['Bucket'] = (df[column1] // bucket_size) * bucket_size
    median_values = df.groupby('Bucket')[column2].mean()

    print(f"{median_values}")
    plt.plot(median_values.index, median_values.values, marker='o', linestyle='-', color='yellow')
    plt.title(f'Median Travel Time vs. Start Time')
    plt.xlabel('Start time (seconds from midnight)')
    plt.ylabel('Travel time (in seconds)')
    plt.grid(True)
    plt.show()

    diagram_file = f'median_travel_time_vs_start_time.png'
    plt.savefig(diagram_file)


def create_travel_time_vs_hour() -> None:
    file_path = '../dijkstra_results.csv'
    df = pd.read_csv(file_path)

    column1 = 'Starting hour'
    column2 = 'Travel time'

    X = df[[column1]]
    y = df[column2]

    plt.scatter(X, y, color='yellow')
    plt.title('Travel Time vs. Start Time')
    plt.xlabel('Start time (seconds from midnight)')
    plt.ylabel(column2)
    plt.legend()
    plt.grid(True)

    diagram_file = 'travel_time_vs_start_time.png'
    plt.savefig(diagram_file)


def create_changes_on_hour_buckets(bucket_size: int = 5) -> None:
    file_path = '../dijkstra_results.csv'
    df = pd.read_csv(file_path)

    column1 = 'Starting hour'
    column2 = 'Travel time'

    df['Bucket'] = (df[column1] // bucket_size) * bucket_size
    median_values = df.groupby('Bucket')[column2].mean()

    print(f"{median_values}")
    plt.plot(median_values.index, median_values.values, marker='o', linestyle='-', color='yellow')
    plt.title(f'Median Number of changes vs. Start Time')
    plt.xlabel('Start time (seconds from midnight)')
    plt.ylabel('Number of changes')
    plt.grid(True)
    plt.show()

    diagram_file = f'median_changes_vs_start_time.png'
    plt.savefig(diagram_file)


def create_result_of_time_buckets(bucket_size: int = 0.01) -> None:
    file_path_model = '../a_star_time_results_model.csv'
    df_model = pd.read_csv(file_path_model)
    file_path_speed = '../a_star_time_results_speed.csv'
    df_speed = pd.read_csv(file_path_speed)
    file_path_mix = '../a_star_time_results_mix.csv'
    df_mix = pd.read_csv(file_path_mix)

    column1 = 'Distance'
    column2 = 'Executing time'

    df_model['Bucket'] = (df_model[column1] // bucket_size) * bucket_size
    df_speed['Bucket'] = (df_speed[column1] // bucket_size) * bucket_size
    df_mix['Bucket'] = (df_mix[column1] // bucket_size) * bucket_size

    median_values_model = df_model.groupby('Bucket')[column2].mean()
    median_values_speed = df_speed.groupby('Bucket')[column2].mean()
    median_values_mix = df_mix.groupby('Bucket')[column2].mean()

    plt.plot(median_values_model.index, median_values_model.values, marker='o', linestyle='-', color='blue')
    plt.plot(median_values_speed.index, median_values_speed.values, marker='o', linestyle='-', color='green')
    plt.plot(median_values_mix.index, median_values_mix.values, marker='o', linestyle='-', color='yellow')
    plt.title(f'Median Executing Time vs Distance')
    plt.xlabel('Distance')
    plt.ylabel('Median Executing Time')
    plt.grid(True)

    diagram_file = f'scatter_a_star_time.png'
    plt.savefig(diagram_file)


def create_result_of_change_buckets(bucket_size: float=0.01) -> None:
    file_path_model = '../a_star_time_results_model.csv'
    df_time = pd.read_csv(file_path_model)
    file_path_change = '../a_star_change_results.csv'
    df_change = pd.read_csv(file_path_change)

    column1 = 'Distance'
    column2 = 'Executing time'

    df_time['Bucket'] = (df_time[column1] // bucket_size) * bucket_size
    df_change['Bucket'] = (df_change[column1] // bucket_size) * bucket_size

    median_values_model = df_time.groupby('Bucket')[column2].mean()
    median_values_speed = df_change.groupby('Bucket')[column2].mean()

    plt.plot(median_values_model.index, median_values_model.values, marker='o', linestyle='-', color='blue')
    plt.plot(median_values_speed.index, median_values_speed.values, marker='o', linestyle='-', color='green')
    plt.title(f'Median Executing Time vs Distance')
    plt.xlabel('Distance')
    plt.ylabel('Median Executing Time')
    plt.legend()
    plt.grid(True)

    diagram_file = f'scatter_a_star_change.png'
    plt.savefig(diagram_file)


if __name__ == "__main__":
    # create_dijkstra_diagram()
    # create_dijkstra_median_diagram()
    # create_dijkstra_diagram_stops()
    # create_dijkstra_median_diagram_stops()
    # create_time_on_distance_buckets()
    # create_travel_time_vs_distance()
    #create_travel_time_vs_hour()
    #create_time_on_hour_buckets()
    #create_result_of_time_buckets()
    create_result_of_change_buckets()
