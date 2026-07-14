# Core MCP 验证记录

任务：`HAL-THESIS-INSTALL-CORE-MCP`

验证原则：只做 STDIO 初始化和只读调用；不执行 Git 写操作，不写入论文数据目录，不下载论文 PDF。

## 配置验证

| 检查 | 结果 |
|---|---|
| `~/.codex/config.toml` TOML 解析 | 通过 |
| 新表重复检查 | 通过，每个新增名称各出现一次 |
| `projectFilesystem` | 保留 |
| `openaiDeveloperDocs` | 保留 |
| Origin | 未添加、未配置 |
| `codex mcp list` | 五个 MCP 均为 enabled |
| `codex mcp get paperSearch` | 工具拒绝列表和超时设置可见 |

最终 MCP 名称：`projectFilesystem`、`openaiDeveloperDocs`、`thesisFilesystem`、`thesisGit`、`paperSearch`。

## thesisFilesystem

启动命令：

```text
npx -y @modelcontextprotocol/server-filesystem@2026.7.10 /mnt/c/Users/Administrator/Desktop/Hal/hal_azi_thesis
```

结果：

- 服务输出 `Secure MCP Filesystem Server running on stdio`。
- MCP `initialize` 和 `tools/list` 成功。
- `list_allowed_directories` 返回的唯一目录是 `/mnt/c/Users/Administrator/Desktop/Hal/hal_azi_thesis`。
- 没有把 `/home`、`/mnt/c`、`/`、`~/.ssh`、`~/.codex` 或 Windows 用户目录整体作为参数传入。
- 写操作仍需要审批；本轮没有调用写工具。

## thesisGit

启动命令：

```text
uvx --from mcp-server-git==2026.7.10 mcp-server-git --repository /mnt/c/Users/Administrator/Desktop/Hal/hal_azi_thesis
```

只读调用结果：

| 工具 | `repo_path` | 结果 |
|---|---|---|
| `git_status` | 论文仓库 | 成功 |
| `git_log` | 论文仓库，`max_count = 1` | 成功 |
| `git_show` | 论文仓库，`revision = HEAD` | 成功 |

没有调用 `git_add`、`git_commit`、`git_reset`、`git_checkout` 或 `git_create_branch`。版本源码包含 `validate_repo_path`，会将请求路径解析后限制在启动参数指定的允许仓库内；因此没有发现可以越过论文仓库的安全失败条件。Git 写工具仍要求确认。

## paperSearch

采用方案 A：

```text
uvx --from paper-search-mcp==0.1.4 paper-search-mcp
```

MCP 初始化和工具列表成功。小型查询为 `cement bond logging acoustic evaluation`，调用顺序优先 Crossref、OpenAlex、Semantic Scholar、arXiv，每个来源请求最多 1 条，最终返回 3 条记录。Semantic Scholar 本次返回 0 条。没有调用 Google Scholar、Sci-Hub、下载或 PDF 阅读工具。

| 标题 | 年份 | DOI 或稳定标识 | 来源 | 开放获取状态 |
|---|---:|---|---|---|
| Operational Guidelines for Supervising Cement Bond Logging and Evaluation | 2025 | `10.1007/978-3-031-95936-3_8` | Crossref | 返回结果未提供 PDF 或 OA 指示 |
| Acoustic Character Logs and Their Applications in Formation Evaluation | 1963 | DOI `10.2118/452-pa`；OpenAlex `W2025893548` | OpenAlex | 返回结果未提供 PDF 或 OA 指示 |
| Yield stress of aerated cement paste | 2019 | arXiv `1910.11845v1` | arXiv | 有 arXiv PDF URL；本轮未访问或下载 |

## 凭据与文件完整性

- `paper-search-mcp` 的可选凭据均为空，未在配置或仓库出现真实值。
- 未创建仓库内 `.env`，未写入 API key、token、邮箱或代理 URL。
- 未发现论文 PDF 下载产物。
- 本轮没有修改论文正文、实验数据、实验指标、`raw`、`data/processed` 或 `results`。

## 会话状态

配置文件已被新的 Codex CLI 进程读取并验证。当前 Codex 会话仍应重启后再使用新增 MCP，状态为 `configuration_valid_restart_required`。
