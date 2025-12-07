import torch
from tqdm import tqdm

class OutputInspector:
  def __init__(self, targetLayer):
      self.layerOutputs = []
      self.featureHandle=targetLayer.register_forward_hook(self.feature)

  def feature(self,model, input, output):
      self.layerOutputs.append(output.detach().cpu())

  def release(self):
      self.featureHandle.remove()

def getActs(model, tokenizer, inputIds):
    model.eval()
    with torch.no_grad():
        if 'GemmaForCausalLM' in str(type(model)) or 'LlamaForCausalLM' in str(type(model)):
            actInspectors = [OutputInspector(layer.mlp.act_fn) for layer in model.model.layers]
        elif 'GPTNeoXForCausalLM' in str(type(model)):
            actInspectors = [OutputInspector(layer.mlp.act) for layer in model.gpt_neox.layers]
        elif 'PhiForCausalLM' in str(type(model)):
            actInspectors = [OutputInspector(layer.mlp.activation_fn) for layer in model.model.layers]
        else:
            print('model is not supported!')

        input_ids = torch.LongTensor([inputIds]).to(model.device)

        outputs = model(input_ids)

        for actInspector in actInspectors:
            actInspector.release()

        acts = torch.cat([torch.cat(actInspector.layerOutputs, dim=1) for actInspector in actInspectors], dim=0).transpose(0,1)

    return acts

def getAveragedActivations(data, model, tokenizer, maxRange=30, position='first'):
    repPosition = '%sPosition'%position

    normalActs = None
    repetiActs = None
    normalTotalPoints = 0
    repetiTotalPoints = 0

    for line in tqdm(data):
        inputIds = line['generatedIds']
        acts = getActs(model, tokenizer, inputIds)
        startingPoint = line[repPosition] - 1
        normalRange = list(range(max(0, startingPoint-maxRange), startingPoint))
        repetiRange = list(range(startingPoint, min(len(inputIds), startingPoint + maxRange)))

        normalTotalPoints += len(normalRange)
        repetiTotalPoints += len(repetiRange)

        na = acts[normalRange].sum(dim=0)
        ra = acts[repetiRange].sum(dim=0)

        if normalActs is None:
            normalActs = na
        else:
            normalActs += na

        if repetiActs is None:
            repetiActs = ra
        else:
            repetiActs += ra

    normalActs /= normalTotalPoints
    repetiActs /= repetiTotalPoints

    return normalActs, repetiActs

def findNeurons(data, model, tokenizer, maxRange=30, position='second'):
    normalActs, repetiActs = getAveragedActivations(data, model, tokenizer, maxRange, position)
    diff = repetiActs - normalActs
    ranks = torch.argsort(diff.flatten(), descending=True)
    width = diff.shape[1]
    sortedNeurons = []
    for r in ranks:
        neuron = (int(r // width), int(r % width))
        info = {
            'neuron': neuron,
            'normalActs':  normalActs[neuron].tolist(),
            'repetitionActs': repetiActs[neuron].tolist(),
            'diffs': diff[neuron].tolist()
        }
        sortedNeurons.append(info)
    return sortedNeurons