# Copyright 2025 CVS Health and/or one of its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Disable MPS on macOS to prevent OOM failures during local test runs.
# (PyTorch will fall back to CPU.)
import torch

if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False
