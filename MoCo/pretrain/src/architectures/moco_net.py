from torch import nn
import torch
from torch import Tensor
from torchvision import models
from collections import OrderedDict
class MoCo_arch(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        self.K = hparams["K"] #negative keys; Queue size
        self.m = hparams["m"] #momentum to update the key enocder
        self.t = hparams["t"] # softmax Temperature

        weights_path = '/home/student/vqa/MOCO/src/architectures'
        #loading pre-trained weights
        pretrained_dict = torch.load(weights_path+'/mimic-chexpert_lr_0.01_bs_128_fd_128_qs_65536.pt')['state_dict']
        state_dict = {}
        for k, v in pretrained_dict.items():
            if k.startswith("model.encoder_q."):
                k = k.replace("model.encoder_q.", "")
                state_dict[k] = v
        if "model.encoder_q.classifier.weight" in pretrained_dict.keys():
            feature_dim = pretrained_dict["model.encoder_q.classifier.weight"].shape[0]
            #in_features = pretrained_dict["model.encoder_q.classifier.weight"].shape[1]
    
            model = models.__dict__['densenet121'](num_classes = feature_dim)
            model.load_state_dict(state_dict)
            # del model.classifier
            # model.add_module("classifier", torch.nn.Linear(in_features, hparams['feature_dim']),)

        else:
            raise RuntimeError("Unrecognized classifier.")
        self.encoder_q = model
        self.encoder_k = models.__dict__[hparams["encoder_K_arch"]](num_classes = hparams['feature_dim'])
        #What is this doing?
        if hasattr(self.encoder_q, "fc"):  # ResNet models
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                
                self.encoder_q.fc = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
                )
                self.encoder_k.fc = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
                )
        elif hasattr(self.encoder_q, "classifier"):  # Densenet models
                dim_mlp = self.encoder_q.classifier.weight.shape[1]
                classifier = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(dim_mlp, 512)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('fc2', nn.Linear(512,256)),
                    ('relu2', nn.ReLU(inplace=True)),
                    ('fc3', nn.Linear(256,hparams['feature_dim']))
                ]))
                self.encoder_q.classifier = classifier
                self.encoder_k.classifier = classifier
                # self.encoder_k.classifier = nn.Sequential(
                #     nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.classifier
                # )
        for q_param, k_param in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k_param.data.copy_(q_param) # initialize parameters of key encoder from q encoder, why?
            k_param.requires_grad = False # no need to update by gradient, because when compared to 2 other contrastive loss mechanisms, updating params of k encoder by momentum is the best


        #create queue registers using pytorch buffers
        self.register_buffer('queue', nn.functional.normalize(torch.randn(hparams["feature_dim"], self.K), dim = 0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))




    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: Tensor):
        # gather keys before updating queue
        #keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        assert isinstance(self.queue_ptr, Tensor)
        ptr = int(self.queue_ptr)
        assert (
            self.K % batch_size == 0
        ), f"batch_size={batch_size}, K={self.K}"  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr 


    def forward(self, im_q: Tensor, im_k: Tensor):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        #negative pair is q and another image randomly chosen right?
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.t

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits.cuda(), labels

#utils
@torch.no_grad()
def concat_all_gather(tensor: Tensor) -> Tensor:
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)

    return output


