class ContextDataset():
    def __init__(self, context_data, x, y, batch_size):
        self.max_index = context_data.shape[0]
        assert(self.max_index == x.shape[0])
        assert(self.max_index == y.shape[0])
        assert (len(context_data.shape) == 2)
        assert(len(x.shape) == 5)
        assert (len(y.shape) == 4)
        self.context_data = context_data
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index+self.batch_size > self.max_index:
            raise StopIteration
        context = self.context_data[self.index:self.index+self.batch_size, :]
        x = self.x[self.index:self.index+self.batch_size, :, :, :, :]
        y = self.y[self.index:self.index+self.batch_size, :, :, :]
        self.index += self.batch_size
        return (context, x), y