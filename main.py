import pandas as pd
import ex1_main


def ask_user_of_tribe() -> bool:
    print("########################################################################")
    choice = input("Exercise 1 or 2? ")
    while choice not in {'1', '2'}:
        print("Just enter '1' or '2'")
        choice = input("Exercise 1 or 2? [1/2]")
    return choice == '1'


def fix_the_data() -> None:
    def time_to_seconds(time_str):
        parts = time_str.split(' ')[1].split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    df = pd.read_csv('dijkstra_results_2.csv')
    df['Starting hour'] = df['Starting hour'].apply(time_to_seconds)

    df.to_csv('dijkstra_results.csv', index=False)


if __name__ == '__main__':
    if ask_user_of_tribe():
        ex1_main.main()
    else:
        ex1_main.handle_user_input()
