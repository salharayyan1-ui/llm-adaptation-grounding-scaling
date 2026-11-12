"""Lightweight NumPy-backed mock for PyTorch to enable testing in host environments without PyTorch.

Provides minimal implementations of torch, torch.cuda, torch.nn, torch.nn.functional,
torch.linalg, and torch.utils.data sufficient for instantiating and verifying models across
Task 1, Task 2, and Task 3.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import sys
import types
import math
import numpy as np


class MockTensor:
    """NumPy-backed tensor mocking torch.Tensor API."""
    
    def __init__(self, data: Any, requires_grad: bool = False, dtype: Any = None, device: Any = None, **kwargs):
        if isinstance(data, MockTensor):
            self._data = data._data.copy()
        elif isinstance(data, np.ndarray):
            self._data = data.copy()
        else:
            if dtype is not None and str(dtype) in ["int64", "torch.long", "long"]:
                dt = np.int64
            elif dtype is not None and str(dtype) in ["bool", "torch.bool"]:
                dt = np.bool_
            else:
                dt = np.float32
            self._data = np.array(data, dtype=dt)
            
        self.requires_grad = requires_grad
        self.device = "cpu"
        self.dtype = self._data.dtype
        
    @property
    def data(self) -> "MockTensor":
        return self
        
    @data.setter
    def data(self, val: Any) -> None:
        if isinstance(val, MockTensor):
            self._data = val._data
        else:
            self._data = np.array(val)
            
    def data_ptr(self) -> int:
        return id(self._data)
        
    def detach(self) -> "MockTensor":
        return MockTensor(self._data, requires_grad=False)
        
    def clone(self) -> "MockTensor":
        return MockTensor(self._data.copy(), requires_grad=self.requires_grad)
        
    def to(self, *args, **kwargs) -> "MockTensor":
        return self
        
    def copy_(self, src: Any) -> "MockTensor":
        s = src._data if isinstance(src, MockTensor) else src
        self._data[...] = s
        return self
        
    def numel(self) -> int:
        return int(self._data.size)
        
    def element_size(self) -> int:
        return int(self._data.itemsize)
        
    def item(self) -> Any:
        return self._data.item()
        
    def __int__(self) -> int:
        return int(self._data.item())
        
    def __float__(self) -> float:
        return float(self._data.item())
        
    def contiguous(self) -> "MockTensor":
        return self
        
    def __getitem__(self, idx: Any) -> "MockTensor":
        res = self._data[idx]
        return MockTensor(res)
        
    def __setitem__(self, idx: Any, val: Any) -> None:
        v = val._data if isinstance(val, MockTensor) else val
        self._data[idx] = v
        
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape
        
    @property
    def ndim(self) -> int:
        return self._data.ndim
        
    def size(self, dim: Optional[int] = None) -> Union[Tuple[int, ...], int]:
        if dim is None:
            return self._data.shape
        return self._data.shape[dim]
        
    def __matmul__(self, other: Any) -> "MockTensor":
        o = other._data if isinstance(other, MockTensor) else other
        return MockTensor(self._data @ o)
        
    def __add__(self, other: Any) -> "MockTensor":
        o = other._data if isinstance(other, MockTensor) else other
        return MockTensor(self._data + o)
        
    def __sub__(self, other: Any) -> "MockTensor":
        o = other._data if isinstance(other, MockTensor) else other
        return MockTensor(self._data - o)
        
    def __mul__(self, other: Any) -> "MockTensor":
        o = other._data if isinstance(other, MockTensor) else other
        return MockTensor(self._data * o)
        
    def __rmul__(self, other: Any) -> "MockTensor":
        return self.__mul__(other)
        
    def __truediv__(self, other: Any) -> "MockTensor":
        o = other._data if isinstance(other, MockTensor) else other
        return MockTensor(self._data / o)
        
    def __pow__(self, power: Any) -> "MockTensor":
        return MockTensor(self._data ** power)
        
    def __neg__(self) -> "MockTensor":
        return MockTensor(-self._data)
        
    def __eq__(self, other: Any) -> "MockTensor":
        o = other._data if isinstance(other, MockTensor) else other
        return MockTensor(self._data == o)
        
    def __ne__(self, other: Any) -> "MockTensor":
        o = other._data if isinstance(other, MockTensor) else other
        return MockTensor(self._data != o)
        
    def __lt__(self, other: Any) -> "MockTensor":
        o = other._data if isinstance(other, MockTensor) else other
        return MockTensor(self._data < o)
        
    def __le__(self, other: Any) -> "MockTensor":
        o = other._data if isinstance(other, MockTensor) else other
        return MockTensor(self._data <= o)
        
    def __gt__(self, other: Any) -> "MockTensor":
        o = other._data if isinstance(other, MockTensor) else other
        return MockTensor(self._data > o)
        
    def __ge__(self, other: Any) -> "MockTensor":
        o = other._data if isinstance(other, MockTensor) else other
        return MockTensor(self._data >= o)
        
    @property
    def T(self) -> "MockTensor":
        return MockTensor(self._data.T)
        
    def numpy(self) -> np.ndarray:
        return self._data
        
    def cpu(self) -> "MockTensor":
        return self
        
    def abs(self) -> "MockTensor":
        return MockTensor(np.abs(self._data))
        
    def max(self) -> "MockTensor":
        return MockTensor(np.max(self._data))
        
    def sum(self, dim: Optional[Any] = None) -> "MockTensor":
        if dim is None:
            return MockTensor(np.sum(self._data))
        return MockTensor(np.sum(self._data, axis=dim))
        
    def view(self, *shape: Any) -> "MockTensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            target_shape = shape[0]
        else:
            target_shape = shape
        return MockTensor(self._data.reshape(target_shape))
        
    def __repr__(self) -> str:
        return f"MockTensor({self._data})"


class MockParameter(MockTensor):
    """NumPy-backed Parameter mocking nn.Parameter."""
    def __init__(self, data: Any, requires_grad: bool = True, **kwargs):
        super().__init__(data, requires_grad=requires_grad, **kwargs)


class MockModule:
    """NumPy-backed Module mocking nn.Module."""
    
    def __init__(self, *args, **kwargs):
        self._modules: Dict[str, Any] = {}
        self._parameters: Dict[str, Any] = {}
        self._buffers: Dict[str, Any] = {}
        self.training = True
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError
        
    def register_buffer(self, name: str, tensor: Any, persistent: bool = True):
        self._buffers[name] = tensor
        super().__setattr__(name, tensor)
        
    def __setattr__(self, name: str, value: Any):
        if isinstance(value, MockParameter):
            if not hasattr(self, "_parameters"):
                super().__setattr__("_parameters", {})
            self._parameters[name] = value
        elif isinstance(value, MockModule):
            if not hasattr(self, "_modules"):
                super().__setattr__("_modules", {})
            self._modules[name] = value
        super().__setattr__(name, value)
        
    def parameters(self):
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()
            
    def named_parameters(self, prefix: str = ""):
        for k, p in self._parameters.items():
            yield f"{prefix}.{k}" if prefix else k, p
        for mk, m in self._modules.items():
            sub = f"{prefix}.{mk}" if prefix else mk
            yield from m.named_parameters(sub)
            
    def modules(self):
        yield self
        for m in self._modules.values():
            yield from m.modules()
            
    def named_modules(self, prefix: str = ""):
        yield prefix, self
        for mk, m in self._modules.items():
            sub = f"{prefix}.{mk}" if prefix else mk
            yield from m.named_modules(sub)
            
    def get_submodule(self, target: str) -> Any:
        if not target:
            return self
        atoms = target.split(".")
        mod = self
        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(f"Module has no attribute {item}")
            mod = getattr(mod, item)
        return mod
        
    def named_buffers(self, prefix: str = ""):
        for k, b in self._buffers.items():
            yield f"{prefix}.{k}" if prefix else k, b
        for mk, m in self._modules.items():
            sub = f"{prefix}.{mk}" if prefix else mk
            yield from m.named_buffers(sub)

    def state_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        for k, p in self.named_parameters():
            res[k] = p.detach().clone()
        for k, b in self.named_buffers():
            res[k] = b.detach().clone()
        return res
        
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        for k, p in self.named_parameters():
            if k in state_dict:
                val = state_dict[k]
                p.data = val._data if isinstance(val, MockTensor) else val
        for k, b in self.named_buffers():
            if k in state_dict:
                val = state_dict[k]
                if isinstance(b, MockTensor):
                    b.data = val._data if isinstance(val, MockTensor) else val
                else:
                    self._buffers[k] = val
        return self
        
    def train(self, mode: bool = True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self
        
    def eval(self):
        return self.train(False)


class MockLinear(MockModule):
    """Mock nn.Linear."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = MockParameter(np.random.randn(out_features, in_features).astype(np.float32))
        self.bias = MockParameter(np.zeros(out_features, dtype=np.float32)) if bias else None
        
    def forward(self, x: Any) -> Any:
        x_data = x._data if isinstance(x, MockTensor) else x
        res = x_data @ self.weight._data.T
        if self.bias is not None:
            res = res + self.bias._data
        return MockTensor(res)


class MockDropout(MockModule):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    def forward(self, x: Any) -> Any:
        return x


class MockIdentity(MockModule):
    def forward(self, x: Any) -> Any:
        return x


class MockGELU(MockModule):
    def forward(self, x: Any) -> Any:
        return x


class MockLayerNorm(MockModule):
    def __init__(self, normalized_shape: Any, **kwargs):
        super().__init__()
        shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.weight = MockParameter(np.ones(shape, dtype=np.float32))
        self.bias = MockParameter(np.zeros(shape, dtype=np.float32))
    def forward(self, x: Any) -> Any:
        return x


class MockSequential(MockModule):
    def __init__(self, *args: Any):
        super().__init__()
        for i, m in enumerate(args):
            setattr(self, str(i), m)
    def forward(self, x: Any) -> Any:
        for m in self._modules.values():
            x = m(x)
        return x


class MockNoGrad:
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass


def setup_mock_torch():
    """Install mock torch in sys.modules if real PyTorch is not available."""
    try:
        import torch
        if not getattr(torch, "__is_mock__", False) and hasattr(torch, "Tensor"):
            return torch
    except ImportError:
        pass
        
    torch = types.ModuleType("torch")
    torch.__is_mock__ = True
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.long = "int64"
    torch.bool = "bool"
    
    def _parse_shape(*shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            return tuple(shape[0])
        return tuple(shape)

    torch.Tensor = MockTensor
    torch.tensor = lambda d, **kw: MockTensor(d, **kw)
    torch.empty = lambda *shape, **kw: MockTensor(np.empty(_parse_shape(*shape), dtype=np.int64 if kw.get("dtype") in ["int64", "torch.long", "long"] else np.float32))
    torch.zeros = lambda *shape, **kw: MockTensor(np.zeros(_parse_shape(*shape), dtype=np.int64 if kw.get("dtype") in ["int64", "torch.long", "long"] else np.float32))
    torch.ones = lambda *shape, **kw: MockTensor(np.ones(_parse_shape(*shape), dtype=np.int64 if kw.get("dtype") in ["int64", "torch.long", "long"] else np.float32))
    torch.full = lambda shape, fill_value, **kw: MockTensor(np.full(_parse_shape(shape), fill_value, dtype=np.int64 if kw.get("dtype") in ["int64", "torch.long", "long"] else np.float32))
    torch.sum = lambda t, **kw: MockTensor(np.sum(t._data if isinstance(t, MockTensor) else t))
    torch.no_grad = MockNoGrad
    
    # torch.cuda
    cuda = types.ModuleType("torch.cuda")
    cuda.is_available = lambda: False
    cuda.device_count = lambda: 0
    cuda.amp = types.ModuleType("torch.cuda.amp")
    cuda.amp.autocast = MockNoGrad
    cuda.amp.GradScaler = lambda *a, **k: None
    torch.cuda = cuda
    
    # torch.linalg
    linalg = types.ModuleType("torch.linalg")
    def mock_svd(A, full_matrices=False):
        arr = A._data if isinstance(A, MockTensor) else A
        u, s, vh = np.linalg.svd(arr, full_matrices=full_matrices)
        return MockTensor(u), MockTensor(s), MockTensor(vh)
    linalg.svd = mock_svd
    torch.linalg = linalg
    
    # torch.nn
    nn = types.ModuleType("torch.nn")
    nn.Module = MockModule
    nn.Parameter = MockParameter
    nn.Linear = MockLinear
    nn.Dropout = MockDropout
    nn.Identity = MockIdentity
    nn.GELU = MockGELU
    nn.LayerNorm = MockLayerNorm
    nn.Sequential = MockSequential
    
    # nn.init
    init = types.ModuleType("torch.nn.init")
    init.kaiming_uniform_ = lambda t, a=math.sqrt(5): t._data.fill(0.01)
    init.zeros_ = lambda t: t._data.fill(0.0)
    nn.init = init
    
    # torch.nn.functional
    F = types.ModuleType("torch.nn.functional")
    def f_linear(input, weight, bias=None):
        w = weight._data if isinstance(weight, MockTensor) else weight
        x = input._data if isinstance(input, MockTensor) else input
        res = x @ w.T
        if bias is not None:
            b = bias._data if isinstance(bias, MockTensor) else bias
            res = res + b
        return MockTensor(res)
    F.linear = f_linear
    
    # torch.utils.data
    data = types.ModuleType("torch.utils.data")
    class MockDataset: pass
    class MockDataLoader: pass
    class MockSampler: pass
    data.Dataset = MockDataset
    data.DataLoader = MockDataLoader
    data.Sampler = MockSampler
    
    utils = types.ModuleType("torch.utils")
    utils.data = data
    
    sys.modules["torch"] = torch
    sys.modules["torch.cuda"] = cuda
    sys.modules["torch.cuda.amp"] = cuda.amp
    sys.modules["torch.linalg"] = linalg
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = F
    sys.modules["torch.nn.init"] = init
    sys.modules["torch.utils"] = utils
    sys.modules["torch.utils.data"] = data
    
    return torch
