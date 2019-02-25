import random
import numpy as np
from functools import reduce


class SetupEpisode:
    def __init__(self, batch_size=600, shot_min=1, shot_max=5, fixed_shot=0, way_min=2, way_max=10000, fixed_way=0):
        super(SetupEpisode, self).__init__()
        assert(way_min) >= 2
        self.batch_size = batch_size
        self.fixed_shot = fixed_shot
        self.shot_min = shot_min
        self.shot_max = shot_max
        self.way_min = way_min
        self.way_max = way_max
        self.fixed_way = fixed_way
        self.current_n_way = -1
        self.current_n_shot = -1
        self.current_n_query = -1

    def create_new_setup(self):
        if self.fixed_way != 0:
            self.current_n_way = self.fixed_way
            if self.fixed_shot == 0:
                self.current_n_shot = random.sample(range(self.shot_min, self.shot_max), 1)[0]
            else:
                self.current_n_shot = self.fixed_shot
            assert(self.batch_size % self.current_n_way == 0)
            self.current_n_query = self.batch_size / self.current_n_way - self.current_n_shot
            self.current_n_query = int(np.round(self.current_n_query))
            assert(self.current_n_way*(self.current_n_shot+self.current_n_query) == self.batch_size)
            # print('\n%d * (%d + %d) = %d' % (self.current_n_way, self.current_n_shot, self.current_n_query, self.batch_size))
        else:
            if self.fixed_shot == 0:
                self.current_n_shot = random.sample(range(self.shot_min, self.shot_max), 1)[0]
            else:
                self.current_n_shot = self.fixed_shot

            way_min_ = self.way_min
            way_max_ = min(self.way_max, np.floor(self.batch_size/(self.current_n_shot + 1)))
            factors = self._find_batch_size_factors(self.batch_size)
            ok_factors = [f for f in factors if way_min_ <= f <= way_max_]
            self.current_n_way = random.sample(ok_factors, 1)[0]

            assert(self.batch_size % self.current_n_way == 0)
            self.current_n_query = self.batch_size / self.current_n_way - self.current_n_shot
            self.current_n_query = int(np.round(self.current_n_query))
            assert(self.current_n_way*(self.current_n_shot+self.current_n_query) == self.batch_size)
            # print('\n%d * (%d + %d) = %d' % (self.current_n_way, self.current_n_shot, self.current_n_query, self.batch_size))

        return self.current_n_way, self.current_n_shot, self.current_n_query

    def get_current_setup(self):
        return self.current_n_way, self.current_n_shot, self.current_n_query

    def _find_batch_size_factors(self, n):
        return set(reduce(list.__add__,
                          ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))
