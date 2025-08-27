import random
import math
import numpy as np

from PSO.PSO import PSO


class PSOMesh(PSO):
    def __init__(self, num_city, data, neighbours_radius, max_iter=20_000):
        super().__init__(num_city, data, max_iter)
        self.neighbours_radius = neighbours_radius
        self.neighbours = self.create_neighbours_topology()

    def create_neighbours_topology(self):
        """
        Creates a neighbor dictionary for a 2D mesh topology.
        Particles are arranged in a 2D grid, and neighbors are defined within `neighbours_radius`.
        """
        neighbor_dict = {}
        grid_size = int(np.ceil(np.sqrt(self.num)))  # Approximate grid dimensions

        for idx in range(self.num):
            x, y = divmod(idx, grid_size)  # 2D coordinates (row, col)
            neighbor_dict[idx] = []

            # Add neighbors within radius
            for dx in range(-self.neighbours_radius, self.neighbours_radius + 1):
                for dy in range(-self.neighbours_radius, self.neighbours_radius + 1):
                    if dx == 0 and dy == 0:
                        continue  # Skip the particle itself
                    nx, ny = x + dx, y + dy  # Neighbor coordinates
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        neighbor_idx = nx * grid_size + ny
                        if neighbor_idx < self.num:  # Check within bounds
                            neighbor_dict[idx].append(neighbor_idx)

        return neighbor_dict

    def get_neighbours_best(self, start_particle_idx):
        """
        Finds the best (minimum-length) neighbor for a given particle based on the mesh topology.
        """
        min_length = self.lenths[start_particle_idx]
        min_index = start_particle_idx

        for neighbor_idx in self.neighbours[start_particle_idx]:
            if self.lenths[neighbor_idx] < min_length:
                min_length = self.lenths[neighbor_idx]
                min_index = neighbor_idx

        return self.particals[min_index]

    def pso(self):
        results = [self.best_l]
        for cnt in range(1, self.iter_max):

            if len(results) > 130 and all(element == results[-1] for element in results[-100:]):
                break
            for i, one in enumerate(self.particals):
                tmp_l = self.lenths[i]

                neighbour_best = self.get_neighbours_best(i)
                new_one, new_l = self.cross(one, neighbour_best)

                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l
                one, tmp_l = self.mutate(one)

                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l

                self.particals[i] = one
                self.lenths[i] = tmp_l
            self.eval_particals()
            if self.global_best_len < self.best_l:
                self.best_l = self.global_best_len
                self.best_path = self.global_best
            if cnt % 20 == 0:
                results.append(self.best_l)
            self.iter_x.append(cnt)
            self.iter_y.append(self.best_l)
        return self.best_l, self.best_path, results
