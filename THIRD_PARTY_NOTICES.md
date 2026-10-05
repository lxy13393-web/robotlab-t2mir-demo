# Third-party notices and publication gate

## RobotLab

The base repository is derived from
[`fan-ziqi/robot_lab`](https://github.com/fan-ziqi/robot_lab). This curated
research-demo repository retains the applicable Apache-2.0 `LICENSE` text but
does not reproduce the upstream Git history. Consult the linked repository for
the authoritative history and upstream source.

## RSL-RL

The vendored RSL-RL subset under
`exts/robot_lab/robot_lab/third_party/rsl_rl/` is distributed under the
BSD 3-Clause License. Its source headers identify the copyright holders as
ETH Zurich and NVIDIA CORPORATION; `modules/normalizer.py` additionally
identifies Preferred Networks, Inc.

```text
BSD 3-Clause License

Copyright (c) 2021 ETH Zurich, NVIDIA CORPORATION
Copyright (c) 2020 Preferred Networks, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## Isaac Lab and Isaac Sim

Isaac Lab and Isaac Sim are NVIDIA projects with their own license and asset
terms. They are runtime dependencies and are not bundled by this release
builder. Generated assets under `local_assets/` are ignored by Git until their
redistribution terms are checked.

## T2MIR-derived implementation

The algorithm is based on *Mixture-of-Experts Meets In-Context Reinforcement
Learning* (NeurIPS 2025):

```bibtex
@inproceedings{wu2025t2mir,
  title={Mixture-of-Experts Meets In-Context Reinforcement Learning},
  author={Wenhao Wu and Fuhong Liu and Haoru Li and Zican Hu and Daoyi Dong and Chunlin Chen and Zhi Wang},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=VMqxRPqdPw}
}
```

The implementation under `methods/t2mir/` is a modified, vendored research
fork rather than an untouched upstream checkout. Its RobotLab integration and
DT2MIR changes are documented in `methods/t2mir/docs/MODIFICATIONS.md`;
upstream identity is documented in `methods/t2mir/docs/UPSTREAM.md`.

**Release gate:** the upstream snapshot used by this project does not contain
an explicit LICENSE file. Renaming or modifying the code does not change that
status. Keep this repository private until the upstream license is confirmed,
or replace the vendored source with a legally distributable patch/reference
workflow.

## Unitree G1 and MuJoCo assets

Robot descriptions, meshes and converted MJCF files may have separate asset
terms. The release builder stores local generated MJCF files in ignored
`local_assets/`; publication requires an explicit asset-license review.
