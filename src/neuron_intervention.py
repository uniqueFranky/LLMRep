from transformers import GenerationConfig
import torch

class Activator():
    def __init__(self, targetLayer, neuronIds, mode, lastN=0):
        self.neuronIds = neuronIds

        assert mode in ['last', 'all', 'lastN'], 'mode should be last or all'
        self.mode = mode
        self.lastN = lastN

        self.outputHandle = targetLayer.register_forward_hook(self.activate)

    def activate(self,model, input, output):
        if self.mode == 'last':
          output[0, -1, self.neuronIds] += 1
        elif self.mode == 'all':
          output[0, :, self.neuronIds] += 1
        elif self.mode == 'lastN':
          output[0, -self.lastN:, self.neuronIds] += 1
        else:
          print(f'{self.mode=} cannot be recognized')
          pass
        return output

    def release(self):
        self.outputHandle.remove()

class Deactivator():
    def __init__(self, targetLayer, neuronIds, mode, lastN=0):
        self.neuronIds = neuronIds

        assert mode in ['last', 'all', 'lastN'], 'mode should be last or all'
        self.mode = mode
        self.lastN = lastN

        self.outputHandle = targetLayer.register_forward_hook(self.deactivate)

    def deactivate(self,model, input, output):
        if self.mode == 'last':
          output[0, -1, self.neuronIds] *= 0
        elif self.mode == 'all':
          output[0, :, self.neuronIds] *= 0
        elif self.mode == 'lastN':
          output[0, -self.lastN:, self.neuronIds] *= 0
        else:
          print(f'{self.mode=} cannot be recognized')
          pass
        return output

    def release(self):
        self.outputHandle.remove()

def convertNeuronsToDict(neurons):
    layer2neurons = {}
    for fn in neurons:
        i, j = fn
        if i not in layer2neurons:
            layer2neurons[i] = []
        layer2neurons[i].append(j)
    return layer2neurons