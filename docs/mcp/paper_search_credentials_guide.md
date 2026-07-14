# Paper Search 凭据指南

当前 `paper-search-mcp` 以无凭据模式运行。Crossref、OpenAlex、Semantic Scholar 和 arXiv 的本轮查询不需要把凭据写入 Codex 配置或论文仓库。

## 外部配置位置

- 配置目录：`/home/dministrator/.config/paper-search-mcp`
- 目录权限：`700`
- 环境文件：`/home/dministrator/.config/paper-search-mcp/.env`
- 文件权限：`600`

当前 `.env` 只有空值模板：

```dotenv
PAPER_SEARCH_MCP_UNPAYWALL_EMAIL=
PAPER_SEARCH_MCP_CORE_API_KEY=
PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY=
PAPER_SEARCH_MCP_ZENODO_ACCESS_TOKEN=
PAPER_SEARCH_MCP_IEEE_API_KEY=
PAPER_SEARCH_MCP_ACM_API_KEY=
```

不要把这些变量的真实值写入：

- 论文仓库的任何文件；
- `~/.codex/config.toml`；
- Git remote URL、命令行参数或提交消息；
- `docs/mcp` 文档、日志或验证输出。

如果未来确实需要增强 API，凭据只能通过用户级外部 secret file 或进程环境注入，并在使用前重新验证日志不会回显值。不要在仓库中创建 `.env`，不要把用户邮箱当作公开配置提交。

## 明确禁止的来源

本配置没有 `PAPER_SEARCH_MCP_GOOGLE_SCHOLAR_PROXY_URL`。Codex 还禁用了 `search_google_scholar`、`search_papers`、`download_scihub` 和 `download_with_fallback`。本项目不使用 Google Scholar 代理或 Sci-Hub。

## 下载边界

本轮不下载论文 PDF。未来如需合法开放获取文件，应先明确来源、输出目录和审批范围，输出目录放在仓库外部，不能使用论文仓库、`results`、`raw` 或 `data/processed`。
