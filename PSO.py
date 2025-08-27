import random
import math
import numpy as np


class PSO(object):
    def __init__(self, num_city, data, max_iter=20_000):
        self.iter_max = max_iter
        self.num = 700
        self.num_city = num_city
        self.location = data
        self.dis_mat = self.compute_dis_mat(num_city, self.location)
        self.particals = self.greedy_init(self.dis_mat,num_total=self.num,num_city =num_city)
        self.lenths = self.compute_paths(self.particals)
        init_l = min(self.lenths)
        init_index = self.lenths.index(init_l)
        init_path = self.particals[init_index]
        self.global_best = init_path
        self.global_best_len = init_l
        self.best_l = self.global_best_len
        self.best_path = self.global_best
        self.iter_x = [0]
        self.iter_y = [init_l]
    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x

                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        return result

    def random_init(self, num_total, num_city):
        tmp = [x for x in range(num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result

    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat

    def compute_pathlen(self, path, dis_mat):
        a = path[0]
        b = path[-1]
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    def eval_particals(self):
        min_lenth = min(self.lenths)
        min_index = self.lenths.index(min_lenth)
        cur_path = self.particals[min_index]
        if min_lenth < self.global_best_len:
            self.global_best_len = min_lenth
            self.global_best = cur_path

    def cross(self, cur, best):
        one = cur.copy()
        l = [t for t in range(self.num_city)]
        t = np.random.choice(l,2)
        x = min(t)
        y = max(t)
        cross_part = best[x:y]
        tmp = []
        for t in one:
            if t in cross_part:
                continue
            tmp.append(t)
        one = tmp + cross_part
        l1 = self.compute_pathlen(one, self.dis_mat)
        one2 = cross_part + tmp
        l2 = self.compute_pathlen(one2, self.dis_mat)
        if l1<l2:
            return one, l1
        else:
            return one, l2

    def mutate(self, one):
        one = one.copy()
        l = [t for t in range(self.num_city)]
        t = np.random.choice(l, 2)
        x, y = min(t), max(t)
        one[x], one[y] = one[y], one[x]
        l2 = self.compute_pathlen(one,self.dis_mat)
        return one, l2

    def pso(self):
        results = [self.best_l]
        for cnt in range(1, self.iter_max):

            if len(results) > 130 and all(element == results[-1] for element in results[-100:]):
                break
            for i, one in enumerate(self.particals):
                tmp_l = self.lenths[i]
                new_one, new_l = self.cross(one, self.global_best)

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

    def run(self):
        best_length, best_path, results = self.pso()
        return self.location[best_path], best_length, results
