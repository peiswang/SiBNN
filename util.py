import torch.nn as nn
import numpy

class BinOp():
    def __init__(self, model):
        # count the number of Conv2d and Linear
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        # find all Conv2d and Linear layers
        self.num_of_params = 0
        self.saved_params = []
        self.target_modules = []

        index = 0
        for m_name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if index !=0 and index != (count_targets-1) and 'fp32' not in m_name:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)
                    self.num_of_params += 1
                index += 1

        # start_range = 1
        # end_range = count_targets-2
        # self.bin_range = numpy.linspace(start_range,
        #         end_range, end_range-start_range+1)\
        #                 .astype('int').tolist()
        # self.num_of_params = len(self.bin_index)
        # self.saved_params = []
        # self.target_modules = []
        # index = -1
        # for m in model.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         index = index + 1
        #         if index in self.bin_range:
        #             tmp = m.weight.data.clone()
        #             self.saved_params.append(tmp)
        #             self.target_modules.append(m.weight)

    def binarization(self):
        #self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.clamp_(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            if len(s) == 4:
                m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            elif len(s) == 2:
                m = self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)
            self.target_modules[index].data.sign_().mul_(m.expand(s))

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])


    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            if len(s) == 4:
                m = weight.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            if len(s) == 4:
                m_add = m_add.sum(3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)
            self.target_modules[index].grad.data = self.target_modules[index].grad.data.mul(1e+9)

    def updateBinaryGradWeight_new(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            '''
            if len(s) == 4:
                m = weight.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s)

            m = m.mul(self.target_modules[index].grad.data)

            w = self.target_modules[index].data.view(-1)
            s = self.target_modules[index].data.size()
            if len(s) == 4:
                mean = 0.
            elif len(s) == 2:
                mean = w.mean(0)
            std = w.std(0)
            '''
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            if len(s) == 4:
                m_add = m_add.sum(3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            m = self.target_modules[index].grad.data
            m[weight.lt(-1.)] = 0
            m[weight.gt(+1.)] = 0
            m.mul_(1.0-1.0/n)
            self.target_modules[index].grad.data = m.add(m_add)
