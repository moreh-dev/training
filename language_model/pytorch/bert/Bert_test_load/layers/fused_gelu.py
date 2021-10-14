# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

#torch._C._jit_set_profiling_mode(False)                                                                                    
#torch._C._jit_set_profiling_executor(False)                                                                                
#torch._C._jit_override_can_fuse_on_cpu(True)                                                                               
#torch._C._jit_override_can_fuse_on_gpu(True) 
# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456

# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
#@torch.jit.script #TUAN
def bias_gelu(bias, y):
    x = bias + y
    return  x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
    #res =  x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
    #return res

bias_gelu_impl = bias_gelu

