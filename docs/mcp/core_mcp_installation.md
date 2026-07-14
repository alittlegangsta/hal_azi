# Core MCP 安装记录

任务：`HAL-THESIS-INSTALL-CORE-MCP`

记录时间：2026-07-14（Asia/Shanghai）

本记录覆盖三个新增 MCP：`thesisFilesystem`、`thesisGit`、`paperSearch`。本轮没有配置、删除或修改 Origin，也没有修改 `projectFilesystem` 或 `openaiDeveloperDocs`。

## 环境与前置条件

| 项目 | 结果 |
|---|---|
| 工作区 | `/mnt/c/Users/Administrator/Desktop/Hal/hal_azi_thesis`，路径核验通过 |
| 分支 | `feature/thesis-depth-blocked-exp007` |
| Node/npm/npx | Node `v22.22.3`，npm/npx `10.9.8` |
| uv/uvx | `0.11.14` |
| Python | `3.12.3` |
| Git | `2.43.0` |
| curl | `8.5.0` |
| jq | 缺失；不阻塞三个 MCP 的安装或 STDIO 验证 |
| Codex CLI | `codex-cli 0.144.4` |

安装前工作区已经有用户未提交修改。本轮没有清理、重置、覆盖或改写这些修改。

## 既有 MCP 保护

安装前和安装后均保留以下配置：

| 名称 | 原有配置摘要 | 处理 |
|---|---|---|
| `projectFilesystem` | `npx -y @modelcontextprotocol/server-filesystem /home/xiaoj/cement-channel-detection` | 原样保留 |
| `openaiDeveloperDocs` | `https://developers.openai.com/mcp` | 原样保留 |

配置文件追加前的备份为：

`/home/dministrator/.codex/backups/config.toml.20260714-153615`

备份文件已验证存在且非空。配置更新采用追加新表的方式，没有覆盖整个 `config.toml`，没有产生重复表，也没有写入凭据。当前 `codex mcp list` 未显示 Origin；本轮完全没有添加或配置 Origin。

## 安装结果

| 名称 | 固定版本 | 实现 | 实际命令 | 作用域 | 状态 |
|---|---|---|---|---|---|
| `thesisFilesystem` | `2026.7.10` | `@modelcontextprotocol/server-filesystem` | `npx -y @modelcontextprotocol/server-filesystem@2026.7.10 /mnt/c/Users/Administrator/Desktop/Hal/hal_azi_thesis` | 仅论文仓库 | 已安装，STDIO 启动和允许目录验证通过 |
| `thesisGit` | `2026.7.10` | `mcp-server-git` | `uvx --from mcp-server-git==2026.7.10 mcp-server-git --repository /mnt/c/Users/Administrator/Desktop/Hal/hal_azi_thesis` | 仅论文仓库 | 已安装，STDIO、只读调用和仓库边界实现检查通过 |
| `paperSearch` | `0.1.4` | `paper-search-mcp` | `uvx --from paper-search-mcp==0.1.4 paper-search-mcp` | 外部学术搜索 API；不指定仓库下载目录 | 已安装，采用方案 A，STDIO 和小型查询通过 |

三个服务均以 `enabled = true`、`required = false` 运行。三项的启动超时分别为 Filesystem/Git 30 秒、Paper Search 60 秒；工具超时分别为 180 秒、180 秒、600 秒；写操作审批模式均为 `writes`。

## 安全配置

`paperSearch` 的 Codex 工具拒绝列表为：

- `search_google_scholar`
- `search_papers`
- `download_scihub`
- `download_with_fallback`

本轮没有配置 Google Scholar 代理，没有启用 Sci-Hub，没有调用下载或读取 PDF 的工具。Git 没有调用 `add`、`commit`、`reset`、`checkout`、创建分支或 tag 的操作。Filesystem 虽然提供写工具，但所有写操作需要 Codex 确认，且服务命令只传入论文仓库路径。

Paper Search 用户配置目录为 `/home/dministrator/.config/paper-search-mcp`，目录权限为 `700`，`.env` 权限为 `600`。`.env` 只包含空值模板，不在 Codex 配置、论文仓库或 Git 历史中保存 API key、token 或邮箱。

## 配置验证与重启状态

Python `tomllib` 解析 `~/.codex/config.toml` 通过，唯一 MCP 名称为：

`openaiDeveloperDocs`、`projectFilesystem`、`thesisFilesystem`、`thesisGit`、`paperSearch`

`codex mcp list` 已识别五个配置。当前已经运行的 Codex 会话不保证动态载入新增服务器，因此进入实际使用前需要重启 Codex 会话，状态记为 `configuration_valid_restart_required`。

## 回滚

若后续发现安装或运行问题，优先使用上述备份恢复原始 `config.toml`，并确认恢复后只剩原有 MCP。若只需移除新增项，则只处理 `thesisFilesystem`、`thesisGit`、`paperSearch`，不触碰 `projectFilesystem`、`openaiDeveloperDocs` 或任何 Origin 条目。回滚不使用 `git reset --hard`，也不触碰论文正文、实验数据、`raw`、`data/processed` 或 `results`。
