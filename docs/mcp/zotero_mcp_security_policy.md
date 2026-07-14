# Zotero MCP 安全策略

## 连接边界

- 只使用 `zotero-mcp-server==0.6.2` 的 `[pdf]` extra。
- 只使用本地 Zotero API：`http://127.0.0.1:23119/api`。
- 只使用 STDIO；不注册 HTTP MCP URL。
- 不安装 Zotero XPI，不使用 23120。
- 不启用 Zotero Web API、WebDAV 或 hybrid write mode。

## 凭据策略

Codex 配置只包含：

```text
ZOTERO_LOCAL=true
```

禁止写入或注入：

- `ZOTERO_API_KEY`
- `ZOTERO_LIBRARY_ID`
- `ZOTERO_LIBRARY_TYPE`
- `ZOTERO_WEBDAV_URL`
- `ZOTERO_WEBDAV_USERNAME`
- `ZOTERO_WEBDAV_PASSWORD`
- OpenAI、Gemini 或 Ollama embedding 凭据

不把 Codex 配置、个人文献正文、附件路径或 API 响应保存到论文仓库。

## 工具策略

Codex 的 `zotero` 条目禁用了 25 个写入或 semantic 更新工具，包括：

- 添加、删除或更新 item；
- 创建、删除或管理 Collection；
- 创建、删除或更新 Note、annotation 和 relation；
- 批量修改 tags/extra；
- 合并重复项；
- semantic search 和 semantic database update。

保留的用途是 collections、关键词搜索、metadata、abstract、children 列表、PDF outline/页面读取和 BibTeX 返回。`default_tools_approval_mode = "writes"` 作为额外防线保留，但写工具已通过 `disabled_tools` 明确禁用。

本轮没有创建 item、添加 DOI、修改标签、创建 Note、修改 Collection、删除/合并重复项或添加附件。

## PDF 与 semantic 边界

`[pdf]` 只提供 PDF outline/annotation 相关能力，不等于允许修改 Zotero 附件。本轮只读取 metadata abstract 和 children 列表，没有读取全文或下载文件。

没有安装 `[semantic]`、`[all]`、ChromaDB 或 sentence-transformers，也没有配置 OpenAI、Gemini 或 Ollama embedding。禁止运行 `zotero-mcp update-db`、`zotero-mcp update-db --fulltext` 或任何 embedding 初始化命令。

## 备份、升级与回滚

配置备份：

```text
/home/dministrator/.codex/backups/config.toml.20260714-161634
```

固定版本维护命令：

```bash
uv tool install --force "zotero-mcp-server[pdf]==0.6.2"
```

配置异常时恢复备份，并只检查 `zotero` 表；不得删除或修改其他 MCP，不得 push。升级前应重新核验 PyPI 和 GitHub release，不自动跟随 latest。
