# EasyR1 for AgentWorld

基于 [EasyR1](https://github.com/hiyouga/EasyR1) 的运行源码，包含 AgentWorld 状态解码与奖励接口。

在工程根目录的独立 RL 环境安装：

```bash
pip install -e .
pip install -e ./EasyR1
pip install -r ./EasyR1/requirements.txt
```

在本目录启动 GRPO：

```bash
python -m verl.trainer.main config=examples/agentworld/grpo.yaml
```

完整数据准备流程见上级目录的 `README.md`。基础版本记录在上级目录的 `FRAMEWORK_VERSIONS.json`；上游源码版权与 Apache-2.0 许可证见 `LICENSE`。
