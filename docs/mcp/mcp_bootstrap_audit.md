# HAL MCP Bootstrap Audit

审计任务：`HAL-THESIS-MCP-BOOTSTRAP-AUDIT`

审计日期：2026-07-14

本轮范围：只调查环境、读取一手安装资料并设计安装方案。未安装软件，未下载可执行文件，未修改 `~/.codex/config.toml`、系统环境、论文正文、原始数据、`data/processed` 或 `results`，未 commit、未 push。

## 1. 当前环境摘要

### 工作区核验

| 检查项 | 实测结果 |
|---|---|
| `pwd` | `/mnt/c/Users/Administrator/Desktop/Hal/hal_azi_thesis` |
| `git rev-parse --show-toplevel` | `/mnt/c/Users/Administrator/Desktop/Hal/hal_azi_thesis` |
| 当前分支 | `feature/thesis-depth-blocked-exp007` |
| remote | `origin https://github.com/alittlegangsta/HAL_EfficientNet`（fetch/push） |
| 工作区 | dirty；`git status --porcelain=v1` 共 191 项，其中 183 项 modified、8 项 untracked，均为本轮之前已存在的状态 |

工作区路径满足任务要求。由于工作区本来就有大量未提交修改，本轮只在新建的 `docs/mcp/` 中写入审计文档，不整理、不覆盖、不回滚其他修改。

### Codex MCP 环境

| 检查项 | 实测结果 |
|---|---|
| `codex --version` | `codex-cli 0.144.4`；命令伴随 PATH alias 只读文件系统 warning，但执行成功 |
| `codex mcp --help` | 支持 `list/get/add/remove/login/logout`；支持 STDIO 命令配置 |
| `codex mcp list` | `projectFilesystem`、`openaiDeveloperDocs` 均为 enabled |
| `test -f ~/.codex/config.toml` | 通过 |
| 配置文件 | 已读取前 240 行；本报告只记录结构和名称，不复制秘密值 |

当前配置中可见的 MCP 表为：

```toml
[mcp_servers.openaiDeveloperDocs]
url = "https://developers.openai.com/mcp"

[mcp_servers.projectFilesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/home/xiaoj/cement-channel-detection"]
```

未在 `~/.codex/config.toml` 或 `codex mcp list` 中发现名为 `Origin` 的条目。用户描述的“Origin MCP 已配置”和当前 WSL 实测状态不一致；这被记录为安装前置核对项，不能据此删除、替换或猜测 Origin 的真实配置。

当前已有的 `projectFilesystem` 使用未固定版本的 `npx -y`，作用域是 `/home/xiaoj/cement-channel-detection`，不是当前 HAL thesis 仓库。后续若要访问 thesis 仓库，应使用新的唯一名称，不覆盖该条目。

### 基础命令

| 命令 | 实测结果 | 安装影响 |
|---|---|---|
| `node` | `/home/dministrator/.nvm/versions/node/v22.22.3/bin/node`, `v22.22.3` | 满足 Node 本地命令需求；Filesystem 上游 package metadata 未声明 `engines` |
| `npm` | `10.9.8` | 可用 |
| `npx` | `10.9.8` | 可用 |
| `uv` | `0.11.14` | 可用 |
| `uvx` | `0.11.14` | 可用 |
| `python3` | `/usr/bin/python3`, `Python 3.12.3` | 满足两个 Python MCP 的 `>=3.10` 要求 |
| `git` | `git version 2.43.0` | 满足 Git MCP 运行需求 |
| `curl` | `8.5.0` | 可用；本轮仅用于只读本地端口探测 |
| `jq` | 未找到，退出码 1 | 非四个目标 MCP 的硬性依赖；安装阶段可不阻塞 |

### Zotero 端口探测

本轮使用 `curl --noproxy '*'` 从 WSL 直接探测：

- `127.0.0.1:23120/mcp`：HTTP 000，连接被拒绝。
- WSL `/etc/resolv.conf` 中的 Windows host gateway `10.255.255.254:23120/mcp`：HTTP 000，连接被拒绝。

这只能说明当前没有可访问的 Zotero MCP 服务，不能区分“插件未安装”“Zotero 未运行”“服务未启用”或“端口已修改”。安装 Zotero 插件后必须从 Windows 和 WSL 分别复测。

## 2. 已安装 MCP 与保护边界

| 名称 | 当前状态 | 类型 | 当前作用域/地址 | 处理原则 |
|---|---|---|---|---|
| `projectFilesystem` | enabled | local STDIO | `/home/xiaoj/cement-channel-detection` | 保留原样，不改名、不删除、不升级、不扩展作用域 |
| `openaiDeveloperDocs` | enabled | remote URL | `https://developers.openai.com/mcp` | 保留原样，不改 URL 或认证行为 |
| `Origin` | 未在当前配置实测发现 | unknown | unknown | 作为保留名保护；在安装前重新核对，不做任何替换操作 |

OpenAI 官方文档说明 Codex CLI、ChatGPT desktop app 和 IDE extension 共享 Codex host 上的 MCP 配置，并支持本地 STDIO 与 Streamable HTTP；因此任何后续配置变更都必须按“加新条目、保留旧条目”的方式进行。[Codex MCP 官方文档](https://developers.openai.com/codex/mcp/)

## 3. Origin MCP 保护策略

1. 本轮不执行 `codex mcp remove`，不修改任何既有表，不复用 `Origin` 这个名称。
2. 安装阶段开始前再次执行 `codex mcp list`，并对 `~/.codex/config.toml` 做脱敏结构检查；如果出现 Origin 条目，逐字保留其 `command`、`args`、`url`、`env_vars`、`env`、`cwd` 和工具策略。
3. 四个新增服务使用唯一名称：`thesisPaperSearch`、`thesisFilesystem`、`thesisGit`、`zotero`。其中 `thesisFilesystem` 不与现有 `projectFilesystem` 合并。
4. 任何回滚只删除本次新增且可识别的条目；不删除或禁用 `Origin`、`projectFilesystem`、`openaiDeveloperDocs`。
5. 若用户确认 Origin 位于另一个 Codex profile、另一个 config 文件或外部插件中，应先记录其真实来源，再继续安装；当前报告不猜测其来源。

## 4. 一手来源核查与推荐实现

核查规则：只使用 OpenAI 官方文档、目标原项目仓库、npm 或 PyPI 页面；未使用 Smithery、Sci-Hub 或 Google Scholar 代理。版本判断以 2026-07-14 可见的官方 release/tag 或 registry package metadata 为准。

| 组件 | 当前官方版本判断 | 包/发行物 | 启动方式 | Transport | Python/Node 要求 | 写能力 | 主要风险与兼容性 |
|---|---|---|---|---|---|---|---|
| Filesystem | release `2026.7.10`；npm 当前版本 `2026.7.10` | `@modelcontextprotocol/server-filesystem` | `npx --yes @modelcontextprotocol/server-filesystem@2026.7.10 <allowed-dir>` | STDIO | package metadata 未声明 `engines`；本机 Node `22.22.3` 可用 | 是：write/edit/create/move/delete 等 | 允许目录内有覆盖、移动和删除；Roots 可能替换命令行目录，必须连接后核验 `list_allowed_directories`。只给最小目录；不要把 `results`、raw 或 `data/processed` 作为写作用域。来源：[README](https://github.com/modelcontextprotocol/servers/blob/main/src/filesystem/README.md)、[npm package](https://www.npmjs.com/package/%40modelcontextprotocol/server-filesystem)、[release 2026.7.10](https://github.com/modelcontextprotocol/servers/releases/tag/2026.7.10) |
| Git | release/PyPI `2026.7.10` | `mcp-server-git` | `uvx --from mcp-server-git==2026.7.10 mcp-server-git --repository <repo>` | STDIO | Python `>=3.10`；本机 Python `3.12.3` | 是：commit/add/reset/create_branch/checkout 等 | 上游标为 early development；工具可直接改变工作树、索引和分支。初始只开放 read/status/diff/log/show/branch 查询工具，并禁用写工具。来源：[README](https://github.com/modelcontextprotocol/servers/blob/main/src/git/README.md)、[PyPI](https://pypi.org/project/mcp-server-git/)、[release 2026.7.10](https://github.com/modelcontextprotocol/servers/releases/tag/2026.7.10) |
| Paper Search | tag/PyPI `v0.1.4` / `0.1.4`，PyPI 发布于 2026-07-02 | `paper-search-mcp` | `uvx --from paper-search-mcp==0.1.4 paper-search-mcp` | 计划按本地命令使用 STDIO；README 未给出独立 HTTP 服务 | Python `>=3.10`；本机 Python `3.12.3` | 是：下载工具会按 `save_path` 写 PDF/文件 | 访问多个外部学术服务；可选 API key/email；源码包含 Google Scholar、Sci-Hub 及统一 `search_papers(sources="all")`。本项目必须禁用 Google Scholar proxy、Sci-Hub 工具及默认全源搜索路径，且初始不把下载目录放入 repo。来源：[README](https://github.com/openags/paper-search-mcp/blob/main/README.md)、[pyproject](https://github.com/openags/paper-search-mcp/blob/main/pyproject.toml)、[PyPI](https://pypi.org/project/paper-search-mcp/)、[tag v0.1.4](https://github.com/openags/paper-search-mcp/tags) |
| Zotero | latest release `v1.5.0`，release asset 为 `zotero-mcp-plugin-1.5.0.xpi` | Windows Zotero plugin XPI；无运行时 npm/PyPI 包 | 在 Zotero Preferences 中启用 integrated server；默认 `23120` | Streamable HTTP | Zotero `>=7.0`；Node `>=18` 仅是插件开发前置，最终用户安装 XPI 不需要 Node | 是：write note/tag/metadata/item，且可在插件设置中禁用 | Windows Zotero 进程监听本地 HTTP；WSL 的 `127.0.0.1` 不一定等于 Windows loopback，需用实际可达 host 地址验证；插件写工具必须先关闭。来源：[README](https://github.com/cookjohn/zotero-mcp/blob/main/README.md)、[release v1.5.0](https://github.com/cookjohn/zotero-mcp/releases/tag/v1.5.0) |

### 版本差异记录

- Filesystem 的 GitHub `main` 中 `src/filesystem/package.json` 仍显示 `0.6.3`，而官方 `servers` release 和 npm registry 已显示 `2026.7.10`。
- Git 的 GitHub `main` 中 `src/git/pyproject.toml` 仍显示 `0.6.2`，而官方 release 和 PyPI 已显示 `2026.7.10`。
- Zotero GitHub `main` 的插件 `package.json` 显示 `1.4.7`，但最新 release asset 是 `1.5.0` XPI。

安装方案使用已发布 registry/release 版本，而不是未同步的 `main` 文件版本。安装阶段应再次核对 registry/release；如果版本发生变化，停止并更新本审计，不使用未固定的 latest。

## 5. 安装命令与配置路径规划（本轮未执行）

OpenAI 官方 CLI 语法是 `codex mcp add <name> -- <stdio-command>`；以下命令仅是下一阶段草案，不是本轮执行记录。[官方配置说明](https://developers.openai.com/codex/mcp/)

### Paper Search

```bash
codex mcp add thesisPaperSearch -- uvx --from paper-search-mcp==0.1.4 paper-search-mcp
```

初始环境不传任何 key、email 或 proxy。若未来确实启用 Unpaywall、CORE 或 Semantic Scholar 增强，只允许通过 WSL 用户环境或用户目录下的外部 secret file 注入变量名，不把值写入仓库、命令历史或 `config.toml`。下载输出固定到 repo 外部的临时/用户目录，例如 `/tmp/hal-paper-search-downloads`，不使用默认的 repo 相对路径。

为满足本项目硬约束，建议首轮 Codex 工具策略禁用：

- `search_google_scholar`
- `search_papers`（其默认 `sources="all"` 会包含 Google Scholar）
- `download_scihub`
- `download_with_fallback`（其默认 `use_scihub=true`）

先使用明确的开放来源专用搜索工具；是否开放 PDF 下载应在单独验证后决定。

### Filesystem

```bash
codex mcp add thesisFilesystem -- npx --yes @modelcontextprotocol/server-filesystem@2026.7.10 /mnt/c/Users/Administrator/Desktop/Hal/hal_azi_thesis/docs/mcp
```

初始允许目录只设为 `/mnt/c/Users/Administrator/Desktop/Hal/hal_azi_thesis/docs/mcp`，用于配置验证和审计文档。建议在 Codex 配置中只启用读取、搜索、列目录和 metadata 工具，禁用 `write_file`、`edit_file`、`create_directory`、`move_file`；连接后立即调用 `list_allowed_directories`，确认 Roots 没有扩展作用域。不能把整个 thesis 根目录作为写入范围，也不能把 `/mnt/c/Users/Administrator/Desktop/Hal/results`、raw、`data/processed` 作为允许写入目录。

### Git

```bash
codex mcp add thesisGit -- uvx --from mcp-server-git==2026.7.10 mcp-server-git --repository /mnt/c/Users/Administrator/Desktop/Hal/hal_azi_thesis
```

初始只启用 `git_status`、`git_diff_unstaged`、`git_diff_staged`、`git_diff`、`git_log`、`git_show` 和只读 branch 查询。禁用 `git_commit`、`git_add`、`git_reset`、`git_create_branch`、`git_checkout`；不向该服务配置 push 权限或任何凭据。工作区已有 191 项 dirty 状态必须作为基线保留，不能在安装验证中重置。

### Zotero

Zotero 不通过 `uvx`、`npx` 或 Codex `add` 安装。下一阶段在 Windows Zotero 中安装 release XPI：

```text
zotero-mcp-plugin-1.5.0.xpi
```

启用插件的 integrated server 后，先从 Windows 本地验证 `http://127.0.0.1:23120/mcp`，再从 WSL 验证。Codex 的 Streamable HTTP 配置应使用 WSL 实际可达的 Windows host 地址；只有 WSL mirrored networking 确认可用时才使用 `127.0.0.1`。安装阶段不写入真实 URL 之前，不执行 Codex 配置变更。

## 6. WSL/Windows 路径规划

| 用途 | WSL/Codex 使用 | Windows/Zotero 使用 | 注意 |
|---|---|---|---|
| 当前 repo | `/mnt/c/Users/Administrator/Desktop/Hal/hal_azi_thesis` | `C:\Users\Administrator\Desktop\Hal\hal_azi_thesis` | STDIO 服务从 WSL 启动时使用左列；不要把 `C:\...` 作为 Linux 命令参数 |
| Filesystem 初始作用域 | `/mnt/c/Users/Administrator/Desktop/Hal/hal_azi_thesis/docs/mcp` | 对应 `C:\Users\Administrator\Desktop\Hal\hal_azi_thesis\docs\mcp` | 只用于验证；不扩展到证据/原始数据目录 |
| Git repository | `/mnt/c/Users/Administrator/Desktop/Hal/hal_azi_thesis` | 同一 Windows 文件系统挂载点 | Git MCP 与 Codex 都在 WSL 中运行，统一使用 WSL 路径 |
| Codex config | `/home/dministrator/.codex/config.toml`（当前 shell 的 home） | 不使用 Windows Codex config | 本轮只读；不要在 repo 复制一份带凭据的配置 |
| Zotero service | WSL 访问 Windows host 的可达地址 | 插件实际监听 Windows 本机端口 `23120` | `127.0.0.1` 语义可能不同；先做端口探测，避免盲填 URL |

`/mnt/c` 文件系统会把 Windows 文件暴露给 WSL 进程。即使 MCP 自身有路径检查，也不应把整个挂载盘、用户 home 或 thesis 根目录交给有写能力的服务。

## 7. Zotero 本地网络连通性风险

Zotero README 宣称 integrated server 使用 Streamable HTTP，默认端口为 `23120`，示例地址为 `http://127.0.0.1:23120/mcp`。该地址在 Windows Zotero 进程内通常表示 Windows loopback；Codex 在 WSL 中使用同一个字符串时可能指向 WSL 自己的 loopback。

当前实测两条路径都没有服务响应：

1. WSL `127.0.0.1:23120`：连接被拒绝。
2. WSL host gateway `10.255.255.254:23120`：连接被拒绝。

因此安装阶段必须按以下顺序验证：

1. Windows Zotero 中确认插件已安装、服务器已启用、端口仍为 `23120`。
2. Windows 端先访问本机 loopback，确认服务返回 MCP 响应而非普通网页或连接错误。
3. WSL 端使用 `curl --noproxy '*'` 测试 WSL loopback 和当前 host gateway。
4. 只在 WSL 能稳定访问后，把实际可达 URL 写入 Codex；不做未经验证的 `0.0.0.0` 暴露、端口转发或防火墙改动。
5. 首次连接只允许 Zotero 查询工具；`write_note`、`write_tag`、`write_metadata`、`write_item` 保持禁用，直到另行确认写入边界。

风险判断是基于当前 WSL/Windows 拓扑和本机探测的工程推断；本轮没有修改 WSL 网络模式、Windows 防火墙、端口转发或系统环境。

## 8. 凭据保存策略

- 本报告、CSV、仓库和 `results` 中不保存 API key、token、邮箱、Bearer 值或其他秘密。
- Paper Search 的基本公开来源不需要 key；Unpaywall 只有在启用时需要有效 email，CORE/Semantic Scholar/Zenodo 等变量是可选或场景相关。由于本项目禁止把值放在仓库，首轮保持未配置。
- 后续若确需凭据，使用 WSL 用户环境、外部用户目录 secret file 或 Codex 的 `env_vars`/环境转发机制；配置中只记录变量名，不记录值。
- 不把凭据放进 `codex mcp add` 命令行，不把它们放进 Git remote URL，不把它们写进 `docs/mcp`，不在日志中回显。
- Zotero README 描述为本地服务，本轮没有发现需要 API key/token 的安装步骤；仍需把 HTTP 服务限制在本机可达范围，不把端口公开到局域网。

## 9. 回滚方案

回滚方案只针对下一阶段新增条目，本轮没有执行：

1. 安装前保存脱敏前的本地 config 备份或由用户自行保留原文件；备份文件不得进入仓库。
2. 记录四个新条目的精确名称和版本；出现问题时只执行对应的 `codex mcp remove <new-name>`，不触碰既有名称。
3. `projectFilesystem`、`openaiDeveloperDocs` 和任何真实存在的 Origin 条目不得通过回滚删除或禁用。
4. 不使用 `git reset --hard`、`git checkout --` 或清理命令恢复工作区；已有 dirty 状态不是安装产物。
5. Zotero 问题通过插件设置禁用 integrated server 或在 Windows Zotero Add-ons 中卸载该 XPI；不删除 Zotero library、PDF 或 thesis 文件。
6. 若仅需停止风险，先将新增服务 `enabled = false` 或禁用其写工具，再分析日志；不修改论文和实验数据。

## 10. 推荐安装顺序

1. **安装前状态复核**：重新确认 `pwd`、Git root、branch、`codex mcp list`，解决 Origin 实测缺失问题；确认不触碰既有条目。
2. **Paper Search**：以 `0.1.4` 固定版本启动 STDIO，不配置凭据，不启用 Google Scholar proxy，不启用 Sci-Hub，不把下载写入 repo。
3. **Filesystem**：以 `2026.7.10` 固定 npm 版本、唯一名称和最小 `docs/mcp` 作用域启动；确认 allowed directories 后只开放读取工具。
4. **Git**：以 `2026.7.10` 固定 PyPI 版本、只绑定当前 repo；先只开放 status/diff/log/show 查询，确认不会 commit、stage、checkout 或创建分支。
5. **Zotero**：Windows 端安装 `v1.5.0` XPI，启用服务并完成 Windows/WSL 双端口验证；先禁用四个写工具。
6. **安装后审计**：运行 `codex mcp list`、脱敏配置检查、各 MCP 最小握手/只读调用和路径边界检查；确认 `results`、raw、`data/processed` 没有新增或修改文件。

## 11. 结论与进入安装阶段的条件

- 基础运行条件：满足。Node/npm/npx、uv/uvx、Python 3.12、Git 和 curl 均可用；`jq` 缺失但不是目标 MCP 的硬依赖。
- 配置安全条件：未完全满足。当前实测没有 Origin 条目，与用户描述不一致；另有一个作用域不同且未固定版本的 `projectFilesystem` 条目。
- Zotero 条件：未满足验证条件。当前端口未监听，且 Windows/WSL 的可达地址尚未确定。
- 建议：下一步可以进入“安装前复核/逐个安装”阶段，但不要在未确认 Origin 真实来源、未固定新增条目名称和未完成 Zotero 网络验证前批量修改 Codex 配置。
