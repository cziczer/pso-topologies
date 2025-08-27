import numpy as np

from PSO.PSO import PSO


class PSORing(PSO):
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
        neighbor_dict = {}
        num_particles = len(self.particals)

        for i in range(num_particles):
            left_index = (i - 1) % num_particles  # Previous particle (wrap around to last if i is 0)
            right_index = (i + 1) % num_particles  # Next particle (wrap around to first if i is last)

            neighbor_dict[i] = {"left": left_index, "right": right_index}

        return neighbor_dict

    def get_neighbours_best(self, start_particle_idx):
        min_length = self.lenths[start_particle_idx]
        min_index = start_particle_idx

        current_node = start_particle_idx

        # Traverse to the left within the given radius
        for _ in range(self.neighbours_radius):
            current_node = self.neighbours[current_node]["left"]
            if self.lenths[current_node] < min_length:
                min_length = self.lenths[current_node]
                min_index = current_node

        current_node = start_particle_idx

        # Traverse to the right within the given radius
        for _ in range(self.neighbours_radius):
            current_node = self.neighbours[current_node]["right"]
            if self.lenths[current_node] < min_length:
                min_length = self.lenths[current_node]
                min_index = current_node

        return self.particals[min_index]
