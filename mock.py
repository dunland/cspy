import time
import random

from cityscopy import Cityscopy


if __name__ == '__main__':
    cityscopy = Cityscopy("settings/cityscopy.json")
    size = cityscopy.table_settings['nrows'] * cityscopy.table_settings['ncols']

    while True:
        grid = [[0, 0] for _ in range(size)]

        for i in range(20):
            grid[int(size * random.random())] = [int(6 * random.random()) - 1, int(5 * random.random()) - 1]

        cityscopy.send_json_to_UDP(grid)

        time.sleep(2)
