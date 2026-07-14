# Zotero MCP STDIO 安装记录

任务：`HAL-THESIS-INSTALL-ZOTERO-MCP-STDIO`

记录日期：2026-07-14（Asia/Shanghai）

## 实现与版本

实现项目：`54yyyu/zotero-mcp`

PyPI 包：`zotero-mcp-server`

固定版本：`0.6.2`

一手资料核验结果：

- [PyPI 项目页](https://pypi.org/project/zotero-mcp-server/) 当前显示 `0.6.2`，要求 Python `>=3.10`，提供 `pdf`、`semantic`、`scite` 和 `all` extras。
- [GitHub README](https://github.com/54yyyu/zotero-mcp) 给出的 CLI 入口是 `zotero-mcp`；本地只读模式只需要 `ZOTERO_LOCAL=true`，不需要 Zotero Web API key 或 Library ID。
- [GitHub releases](https://github.com/54yyyu/zotero-mcp/releases) 当前最新 release/tag 为 `v0.6.2`。
- README 将 `[pdf]` 用于 PDF outline/EPUB annotation 支持；本轮没有安装 `[all]` 或 `[semantic]`。

## 安装过程

安装前发现 uv tool 已有 `zotero-mcp-server v0.6.2`，但检查不到 `fitz` 模块，说明原环境不是本轮要求的 `[pdf]` 安装。随后执行固定版本重装：

```bash
uv tool install --force "zotero-mcp-server[pdf]==0.6.2"
```

验证结果：

- 包版本：`0.6.2`。
- `fitz` 模块：存在，`[pdf]` 已安装。
- `chromadb`：不存在。
- `sentence_transformers`：不存在。
- 未运行 `zotero-mcp setup`，没有自动修改 Claude Desktop 或 Claude Code 配置。

实际入口：

```text
/home/dministrator/.local/bin/zotero-mcp
```

解析后的 uv tool 入口：

```text
/home/dministrator/.local/share/uv/tools/zotero-mcp-server/bin/zotero-mcp
```

`zotero-mcp --help` 和 `zotero-mcp setup-info` 均执行成功。

## Transport 与 Codex 注册

Transport 为本地 STDIO，不使用 HTTP MCP URL：

```toml
[mcp_servers.zotero]
command = "/home/dministrator/.local/bin/zotero-mcp"
args = []
enabled = true
required = false
startup_timeout_sec = 30
tool_timeout_sec = 300
default_tools_approval_mode = "writes"

[mcp_servers.zotero.env]
ZOTERO_LOCAL = "true"
```

`zotero` 条目在本轮开始时已经存在，因此没有重复添加。配置更新前备份为：

`/home/dministrator/.codex/backups/config.toml.20260714-161634`

TOML 解析通过；`projectFilesystem`、`openaiDeveloperDocs`、`thesisFilesystem`、`thesisGit` 和 `paperSearch` 的解析值与备份一致；没有添加 Origin。

## Zotero API 目标

连接地址固定为：

```text
http://127.0.0.1:23119/api
```

本轮没有使用 Zotero Web API key、Library ID、Library Type、WebDAV 凭据或 embedding 凭据，也没有使用 23120 端口或安装 Zotero XPI 插件。

## 升级与回滚

固定版本升级或重装命令：

```bash
uv tool install --force "zotero-mcp-server[pdf]==0.6.2"
```

配置回滚应恢复本轮备份：

```bash
cp /home/dministrator/.codex/backups/config.toml.20260714-161634 ~/.codex/config.toml
```

回滚只针对 Codex 用户配置，不触碰 Zotero 文献库、论文仓库或其他 MCP。
