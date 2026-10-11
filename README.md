# minROP

minROP 1.0 是一个用于 ELF64 x86-64 二进制的轻量级 ROP gadget 分析与有限约束搜索框架。它将本地 ELF 扫描结果或用户提供的机器码片段转换为现有指令模型，通过符号执行和 Z3 判断初始条件与最终目标能否同时成立。

项目用于教学与有限范围分析。它不运行目标程序，不生成 exploit 或可部署的 payload，也不提供完整 ROP 框架或 angr 的全程序分析能力。

## 功能

- ELF64 x86-64 gadget extraction：扫描可执行段中的 RET 结尾片段。
- Supported instruction filtering：显式过滤现有语义模型不支持的片段。
- Gadget JSON export：保存地址、完整机器码、指令顺序和唯一 ID。
- Symbolic execution：建模寄存器、部分寄存器、内存、栈和 EFLAGS。
- SMT constraint solving：复用 Z3 求解共同的初始条件、执行有效性与最终目标。
- Finite sequence search：有限候选、有界深度、稳定遍历顺序与可选基础剪枝。

## 安装

需要 Python **3.10 或更新版本**。依赖由 `pyproject.toml` 管理：`capstone`、`pyelftools`、`z3-solver`；测试额外依赖 `pytest`。

将 `<repository-url>` 替换为实际 GitHub 仓库地址：

```text
git clone <repository-url> minROP
cd minROP
python -m venv .venv
```

激活虚拟环境：Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1`，Linux/macOS 使用 `source .venv/bin/activate`。然后安装项目：

```text
python -m pip install -e .
```

安装后可使用 `minirop`，也可使用等价的 `python -m minirop`。版本查询：

```text
minirop --version
```

输出 `minirop 1.0`。Windows 如需将下载缓存和测试临时文件留在项目所在磁盘，可在安装或测试前设置：

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TEMP = (Join-Path $PWD '.tmp')
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = (Join-Path $PWD '.tmp/pip-cache')
$env:PYTHONDONTWRITEBYTECODE = '1'
```

## 使用

从项目根目录执行下面的通用 JSON 示例，不需要额外二进制文件。

### 固定序列求解

```text
minirop solve examples/solve.json
```

该示例读取初始内存中的 6，加 1，并使用给定的栈返回地址结束。预期输出包含 `FOUND`、`SAT` 和 `Selected candidate IDs: ["load_add"]`。

### 有限候选搜索

```text
minirop search examples/search.json
```

该示例提供加 1 和加 2 两个片段，初始 RAX 为 0，目标 RAX 为 2。预期命中 `add_two`，状态为 `FOUND / SAT`。

可读结果视图保留原始 SMT witness，并额外显示选中片段及已求出的初始寄存器、栈与内存值：

```text
minirop search examples/search.json --view payload
```

默认 `--view raw` 保持原始输出。可读视图只格式化已有模型，缺失值不补零；栈偏移相对于初始 RSP，内存表示初始 M0。它不推导 ABI、调用约定或栈布局，因此会明确显示 `Payload reconstruction unavailable.`。

### 扫描用户提供的 ELF

以下命令中的 `path/to/input.elf` 是用户自己的 ELF64 x86-64 文件，不是仓库内置示例：

```text
minirop gadgets path/to/input.elf
minirop gadgets path/to/input.elf -o gadgets.json --supported-only
```

第一条命令输出文本。`-o` / `--output` 输出 UTF-8 JSON；不指定 `--supported-only` 时导出所有扫描结果，指定时只导出现有指令语义检查接受的片段，并报告排除数量。该过滤不保证任意初始状态下都能正常执行。

JSON 包含 `input_version: 1`、`architecture: "x86_64"`、`source` 和 `gadgets`。每个 gadget 保存 `id`、十六进制 `address`、有序 `instructions` 及十六进制 `bytes`。`source` 统一使用 `/` 分隔符，保留相对或绝对路径形式，仅记录来源。搜索读取 JSON 时不会重新打开 ELF。

搜索条件独立保存在用户创建的 `search_config.json`，例如：

```json
{
  "input_version": 1,
  "initial": {
    "registers": {"rsp": "0x8000"},
    "stack": [7, "0x9000"]
  },
  "goals": {"rdi": 7, "rip": "0x9000"},
  "max_depth": 1,
  "pruning": false
}
```

按实际候选及目标调整配置后运行：

```text
minirop search gadgets.json --config search_config.json
```

扫描文件本身没有最终目标或搜索深度，必须在独立配置中提供，或直接在 gadget JSON 中加入 `goals` 与 `max_depth`。重复条件或不支持的指令会明确报错，不会静默覆盖或跳过。完整 ELF demo 后续单独提供；本仓库不包含第三方 CTF 二进制或本地生成的扫描结果。

### 输入格式

`solve` / `search` 接受 UTF-8 JSON，使用 `input_version: 1`。通用示例中的 `candidates` 是非空有序数组，每项包含唯一 `id`、64 位入口 `address` 和完整机器码 `bytes`。字节必须完整解码、满足现有合法操作数检查，并恰好以一个 RET 结束。

| 字段 | 含义 |
| --- | --- |
| `initial.registers` | 初始寄存器等式条件，支持现有 8/16/32/64 位别名。 |
| `initial.flags` | 可选 CF、ZF、SF、OF、PF、AF 布尔条件。未定义标志位不能当作零。 |
| `initial.memory` | 初始 M0 中的小端 64 位字，如 `[{"address": "0x2000", "value": 6}]`。 |
| `initial.stack` | 初始 RSP、RSP+8 等位置的 64 位字，按数组顺序绑定。 |
| `goals` | 必填、非空的最终寄存器等式目标。 |
| `sequence` | `solve` 必填的非空候选 ID 数组，按顺序执行。 |
| `max_depth` | `search` 必填的正整数，枚举深度 1 到该上限。 |
| `pruning` | 可选布尔值，默认 false，启用已有基础剪枝。 |
| `check_entry` | 可选布尔值，默认 false，检查初始 RIP 与首片段入口。 |
| `timeout_ms` | 每次求解超时，默认 5000，范围 1 到 2³¹−1。 |

数值可使用 JSON 整数或十进制/十六进制整数字符串。负值按对应宽度的补码解释，超范围值拒绝。缺省寄存器和栈输入保持符号状态；初始条件、内存别名、返回目标及最终目标在同一求解上下文中检查。未知字段、重复 JSON 键、无效 ID、损坏编码和不支持的指令明确拒绝。

## 理解结果

| STATUS | Solver status | Goals satisfied | 含义 |
| --- | --- | --- | --- |
| FOUND | SAT | True | 在给定候选和深度范围内找到满足共同约束的正常执行序列。 |
| NOT_FOUND | UNSAT | False | 固定序列不可满足，或有限范围内所有候选已确定不可满足；不表示全局无解。 |
| INCONCLUSIVE | UNKNOWN | Unknown | 存在 UNKNOWN、未完成求解、执行有效性失败或资源限制，不能宣称无解。 |

输出还包含选中候选 ID、序列深度、已检查候选数、最终目标值和一个初始状态模型。模型只是一个满足约束的见证，不保证唯一。FOUND / NOT_FOUND / INCONCLUSIVE 的退出码分别是 0 / 1 / 3；分析输入错误返回 2 并写入标准错误，不伪装成普通 UNSAT。`gadgets` 无法读取合法 ELF 时返回 1。

## 支持范围与限制

主要支持数据搬运和寻址、部分寄存器、整数算术与逻辑、乘除、移位旋转、整数辅助指令、栈、EFLAGS、CMP/TEST、SETcc 和寄存器 CMOVcc。Python API 还支持有界固定程序中的直接 Jcc。

CLI 的求解输入聚焦 RET 结尾片段的固定序列和有限候选搜索；Jcc 路径执行需要 Python API。除零和商溢出不作为合法 SAT 执行。POPCNT 仅建模指令语义，不模拟完整 CPUID。

不覆盖完整 x86-64 ISA，不支持 SIMD、浮点、系统特权指令、SHLD/SHRD、BMI、复杂 REP 字符串或完整异常处理。部分栈、内存 CMOV/XCHG 及特殊前缀形式仍不支持。输入仅表达常量等式与 64 位初始内存字，不接受任意 Python/Z3 表达式。

不进行搜索过程中自动扩充候选、跨程序控制流恢复、复杂循环分析、全局最优搜索或真实内存映射/权限验证。ELF 导出保留文件虚拟地址，不自动处理装载偏移、ASLR、libc 或远程目标。工具不会生成或执行利用载荷。

## 测试

安装测试依赖并从项目根目录运行：

```text
python -m pip install -e ".[test]"
pytest
```

也可使用 `python -m pytest`。如需显式限制临时目录到项目中，使用 `pytest --basetemp=.tmp/pytest`；Windows 的缓存与临时目录设置见安装部分。

测试通过代码生成最小 ELF 和临时 JSON 文件，不依赖外部编译器、CTF 二进制或本地扫描产物。通用 `solve.json` / `search.json` 示例及 ELF → JSON → search → Z3 流程均由端到端测试验证。

## 仓库结构

```text
minROP/
├── minirop/
├── tests/
├── examples/
│   ├── solve.json
│   └── search.json
├── README.md
├── LICENSE
├── pyproject.toml
└── .gitignore
```

## 许可证

项目采用 [MIT License](LICENSE)。
