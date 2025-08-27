import random
import math
import numpy as np

from PSO.PSO import PSO

class PSOTree(PSO):
    def __init__(self, num_city, data, neighbours_radius, max_iter=20_000):
        super().__init__(num_city, data, max_iter)
        self.neighbours_radius = neighbours_radius
        self.neighbours = self.create_neighbours_topology()

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

                if new_l < tmp_l or np.random.rand()<0.1:
                    one = new_one
                    tmp_l = new_l
                one, tmp_l = self.mutate(one)

                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand()<0.1:
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


    def create_neighbours_topology(self):
        """
                Creates a neighbor dictionary for a tree topology from the list of particles.

                :return: A neighbor dictionary where each particle index maps to its parent, left, and right children.
                """
        neighbor_dict = {}
        num_particles = len(self.particals)

        for i in range(num_particles):
            # Calculate indices for left and right children
            left_index = 2 * i + 1
            right_index = 2 * i + 2

            # Initialize the node entry in the neighbor dictionary
            neighbor_dict[i] = {
                "parent": (i - 1) // 2 if i > 0 else None,
                "left": left_index if left_index < num_particles else None,
                "right": right_index if right_index < num_particles else None
            }
        return neighbor_dict

    def get_neighbours_best(self, start_particle_idx):
        min_length = self.lenths[start_particle_idx]
        min_index = start_particle_idx

        queue = [(start_particle_idx, 0)]  # (current_node, current_depth)
        visited = set()

        while queue:
            current_node, current_depth = queue.pop(0)
            visited.add(current_node)

            # Update min_length and min_index if the current node's length is smaller
            if self.lenths[current_node] < min_length:
                min_length = self.lenths[current_node]
                min_index = current_node

            if current_depth < self.neighbours_radius:
                parent = self.neighbours[current_node]["parent"]
                left = self.neighbours[current_node]["left"]
                right = self.neighbours[current_node]["right"]

                if parent is not None and parent not in visited:
                    queue.append((parent, current_depth + 1))
                if left is not None and left not in visited:
                    queue.append((left, current_depth + 1))
                if right is not None and right not in visited:
                    queue.append((right, current_depth + 1))

        return self.particals[min_index]

    def run(self):
        best_length, best_path, results = self.pso()
        return self.location[best_path], best_length, results
