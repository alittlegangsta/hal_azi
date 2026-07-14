# Zotero MCP 连接审计

任务：`HAL-THESIS-INSTALL-ZOTERO-MCP-STDIO`

本审计只保存连接状态、工具计数和成功/失败状态，不保存个人文献标题、作者、正文、附件路径或 BibTeX 内容。

## Windows Zotero

| 检查 | 结果 |
|---|---|
| Zotero 进程 | 正在运行 |
| 端口 | `127.0.0.1:23119` 正在监听 |
| Windows 本机 API | HTTP `200`，`application/json` |
| Zotero API | API v3；Zotero `9.0.6` |
| 响应正文 | 未记录，避免保存个人文献内容 |

没有修改 Zotero 设置。没有访问 23120，也没有安装 `cookjohn/zotero-mcp` XPI。

## WSL 连通性

WSL 对 `http://127.0.0.1:23119/api/users/0/items?limit=1` 收到 HTTP `200` 和 `application/json` 头部，说明 WSL 可以访问 Windows Zotero 本地 API。

Zotero 保持 HTTP keep-alive。第一次使用默认 curl 参数时，curl 在收到响应数据后等待连接关闭并返回退出码 28；这不是 connection refused 或 timeout before response。使用仅输出响应头、抑制正文的方式确认已收到 API 响应。没有修改防火墙、portproxy、WSL 网络模式或公网转发。

## STDIO MCP 验证

启动命令：

```text
ZOTERO_LOCAL=true /home/dministrator/.local/bin/zotero-mcp
```

验证均在短生命周期进程中完成：

| 操作 | 结果 |
|---|---|
| MCP `initialize` | 成功 |
| 工具列表 | 61 个工具；包含所需只读工具 |
| 获取 collections | 成功；只记录成功状态，不记录集合名称 |
| 搜索 `acoustic logging` | 成功，最多返回 3 条 |
| 读取一条 metadata | 成功，abstract 字段存在 |
| 列出该条目 children | 成功；不记录附件路径或正文 |
| 导出一条 BibTeX | 成功；仅返回最终回复，不写入 `references.bib` |
| 写工具 | 未调用 |
| semantic 工具 | 未调用；semantic extra 未安装 |

读取 abstract 已满足附件场景的最小内容验证；本轮没有读取或导出 PDF 全文。

## 配置状态

Codex 当前配置已被新的 CLI 进程解析，`zotero` 为 enabled、STDIO、绝对命令路径、`ZOTERO_LOCAL=true`，并设置 30 秒启动超时、300 秒工具超时和 `writes` 审批模式。当前正在运行的 Codex 会话不保证动态刷新配置，状态记为：

`configuration_valid_restart_required`

## 失败处理

若未来 WSL 无法访问 23119，不应修改防火墙、创建公网转发或写入错误 URL。安全替代方案是：在 Windows 运行同一 STDIO 服务，或另行评估 Zotero Web API 只读凭据；本轮没有配置后者。
