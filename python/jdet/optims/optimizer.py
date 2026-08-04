from jdet.utils.registry import OPTIMS

from jittor import optim
import jittor as jt


class Optimizer(object):
    def parameters(self):
        """Return serializable optimizer state without live model references.

        Jittor Adam/AdamW store first and second moments inside param_groups.
        Omitting the groups silently restarted the optimizer after resume.
        """
        data = {}
        for k, d in self.__dict__.items():
            if k in ("param_groups", "_grad_map"):
                continue
            data[k] = d
        data["param_groups"] = []
        for group in self.param_groups:
            state = {k: v for k, v in group.items()
                     if k not in ("params", "grads")}
            # jt.save may replace nested Vars with NumPy arrays in-place.
            # Snapshot moments so saving cannot mutate a running optimizer.
            for state_key in ("m", "values"):
                if state_key in state:
                    state[state_key] = [v.numpy().copy()
                                        for v in state[state_key]]
            data["param_groups"].append(state)
        return data

    def load_parameters(self, data):
        if isinstance(data, dict):
            for k, d in data.items():
                if k == "param_groups":
                    if len(d) != len(self.param_groups):
                        raise ValueError(
                            f'optimizer group count mismatch: '
                            f'{len(d)} vs {len(self.param_groups)}')
                    for saved, current in zip(d, self.param_groups):
                        for state_key, state_value in saved.items():
                            if state_key in ("params", "grads"):
                                continue
                            if state_key in ("m", "values"):
                                if len(state_value) != len(current["params"]):
                                    raise ValueError(
                                        f'optimizer {state_key} length mismatch: '
                                        f'{len(state_value)} vs '
                                        f'{len(current["params"])}')
                                for dst, src in zip(current[state_key],
                                                    state_value):
                                    if not isinstance(src, jt.Var):
                                        src = jt.array(src)
                                    dst.update(src)
                            else:
                                current[state_key] = state_value
                elif k in self.__dict__:
                    self.__dict__[k] = d

    def cur_lr(self):
        return self.param_groups[0].get("lr", self.lr)


@OPTIMS.register_module()
class SGD(optim.SGD, Optimizer):
    def __init__(self, params, lr, momentum=0, weight_decay=0, dampening=0, nesterov=False, grad_clip=None):
        super(SGD, self).__init__(params, lr, momentum, weight_decay, dampening, nesterov)
        self.grad_clip = grad_clip

    def pre_step(self, loss, retain_graph=False):
        super(SGD, self).pre_step(loss)
        if self.grad_clip is not None:
            self.clip_grad_norm(**self.grad_clip)


@OPTIMS.register_module()
class GradMutilpySGD(optim.SGD, Optimizer):
    def __init__(self, grad_clip=None, **kwargs):
        super(GradMutilpySGD, self).__init__(**kwargs)
        self.grad_clip = grad_clip

    def step(self, loss):
        if loss is not None:
            self.pre_step(loss)
        if self.grad_clip is not None:
            self.clip_grad_norm(**self.grad_clip)
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            momentum = pg.get("momentum", self.momentum)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            dampening = pg.get("dampening", self.dampening)
            nesterov = pg.get("nesterov", self.nesterov)

            m = pg.get("grad_mutilpy", 1)
            # optimize main body
            for p, g, v in zip(pg["params"], pg["grads"], pg["values"]):
                if p.is_stop_grad(): continue
                dp = p * weight_decay + g * m
                v.update(momentum * v + dp * (1 - dampening))
                if nesterov:
                    p.update(p - (dp + momentum * v) * lr)
                else:
                    p.update(p - v * lr)
        self.zero_grad()


@OPTIMS.register_module()
class Adam(optim.Adam, Optimizer):
    pass


@OPTIMS.register_module()
class AdamW(optim.AdamW, Optimizer):
    def __init__(self, params, lr, eps=1e-8, betas=(0.9, 0.999), weight_decay=0, grad_clip=None):
        super(AdamW, self).__init__(params, lr, eps, betas, weight_decay)
        self.grad_clip = grad_clip

    def pre_step(self, loss, retain_graph=False):
        super(AdamW, self).pre_step(loss, retain_graph)
        if self.grad_clip is not None:
            self.clip_grad_norm(**self.grad_clip)
