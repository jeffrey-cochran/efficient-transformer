import numpy.random as npr
import torch.utils.data as data


class CustomSampler(data.sampler.Sampler):
    def __init__(self, index_list, shuffle, wrap):
        self.index_list = index_list
        self.shuffle = shuffle
        #
        # if(wrap) ==> do not call StopIteration exception
        # wrap==True used during training, and wrap==False used during test.
        self.wrap = wrap
        self._reset_iter()

    def __iter__(self):
        return self

    def __next__(self):

        wrapped = False

        #
        # Check whether exhausted indices
        if self.iter_counter == len(self._index_list):
            self._reset_iter()
            #
            # Wrap around to form new batch if training
            if self.wrap:
                wrapped = True
            else:
                raise StopIteration()
        #
        # NOTE: if(self.shuffle) ==> index is random
        # elem = ( index, current_iteration, has_wrapped )
        elem = (self._index_list[self.iter_counter], self.iter_counter + 1, wrapped)
        #
        self.iter_counter += 1
        return elem

    def next(self):
        return self.__next__()

    def _reset_iter(self):
        #
        # Reshuffle indices
        if self.shuffle:
            rand_perm = npr.permutation(len(self.index_list))

            #
            # _index_list is just shuffled index_list here
            self._index_list = [self.index_list[_] for _ in rand_perm]
        else:
            self._index_list = self.index_list

        self.iter_counter = 0

    def __len__(self):
        return len(self.index_list)

    def load_state_dict(self, state_dict=None):
        #
        # Why???
        if state_dict is None:
            return
        #
        # State is characterzized by just these two things
        # list of (random) indices and the current iteration
        self._index_list = state_dict["index_list"]
        self.iter_counter = state_dict["iter_counter"]

    def state_dict(self, prefetched_num=None):
        #
        # Prefetched num indicates how many have already been sampled?
        prefetched_num = prefetched_num or 0
        return {
            "index_list": self._index_list,
            "iter_counter": self.iter_counter - prefetched_num,
        }
