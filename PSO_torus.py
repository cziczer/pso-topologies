import numpy as np

from PSO.PSO import PSO

class PSOTorus(PSO):
    def __init__(self, num_city, data, neighbours_radius, max_iter=20_000, grid_width=80):
        super().__init__(num_city, data, max_iter)
        self.neighbours_radius = neighbours_radius
        self.grid_width = grid_width
        self.neighbours = self.create_neighbours_topology()

    def create_neighbours_topology(self):
        """
        Creates a torus topology where particles are arranged in a grid, with wrap-around connections.
        """
        neighbor_dict = {}
        for row in range(self.grid_width):
            for col in range(self.grid_width):
                idx = row * self.grid_width + col
                left = row * self.grid_width + (col - 1) % self.grid_width
                right = row * self.grid_width + (col + 1) % self.grid_width
                up = ((row - 1) % self.grid_width) * self.grid_width + col
                down = ((row + 1) % self.grid_width) * self.grid_width + col
                neighbor_dict[idx] = [left, right, up, down]
        return neighbor_dict

    def get_neighbours_best(self, start_particle_idx):
        """
        Finds the best (minimum-length) neighbor for a given particle based on the torus topology,
        restricted by the specified neighbours_radius.
        """
        def bst_traverse(node_idx, depth, visited):
            nonlocal min_length, min_index

            # Stop traversal if depth exceeds the radius or node is already visited
            if node_idx is None or depth > self.neighbours_radius or node_idx in visited:
                return

            # Mark the current node as visited
            visited.add(node_idx)

            # Update the best neighbor if the current node is better
            if self.lenths[node_idx] < min_length:
                min_length = self.lenths[node_idx]
                min_index = node_idx

            # Perform a recursive traversal on the neighbors
            for neighbor in self.neighbours[node_idx]:
                if neighbor not in visited:
                    bst_traverse(neighbor, depth + 1, visited)

        # Initialize the traversal
        min_length = self.lenths[start_particle_idx]
        min_index = start_particle_idx
        visited = set()

        # Start traversal from the root node with depth 0
        bst_traverse(start_particle_idx, depth=0, visited=visited)

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
