# astrbot_plugin_llm_manager · LLM 供应商管理

把 AstrBot 的供应商配置收敛为**一个虚拟 Provider**，按三级结构统一管理：

```
第一级  API Base URL（base_url + api_key + 站名）
第二级  实例唯一 ID（自动生成：站名_分组名）
第三级  模型 ID（带全局序号 [1][2][3]…，切换用序号即可）
```

**解决的核心痛点**：新增一个模型/分组不再需要新增一个 AstrBot 供应商；供应商多起来后，`/llm use 3` 一条指令全局切换。

---

## 安装

> ⚠️ **顺序很重要**：先 `/llm import` 导入（此时 cmd_config.json 里还是你原来的默认模型），再去 WebUI 把 default_provider_id 改成 LLM Manager。如果先改了 default_provider_id，原来的默认模型值就丢失了，导入时无法自动保留。

1. 把 `astrbot_plugin_llm_manager` 文件夹放入 AstrBot 的插件目录（`data/plugins/`，或用 WebUI 插件管理导入）。
2. **重启 AstrBot**（或重载插件）。启动后插件会自动注册一个新提供商类型 **LLM Manager**。
3. 在 QQ 里发 **`/llm import`**，从系统配置一键导入已有供应商和模型。导入时会自动读取你原来的 `default_provider_id` 并设为插件全局默认，确保安装后默认模型不变。
4. 打开 WebUI → **服务提供商**：
   - 新增提供商：类型选 **LLM Manager**，`id` 随意（如 `llm_manager`），`key` 留空，保存。
   - 在该提供商的「模型」区域点「自定义模型」，添加一个占位模型，模型名填 **`main`**（**不要填真实模型名**），保存。
5. 左侧菜单 **配置** → 找到 **默认聊天模型** → 选择 `llm_manager / main` → 保存配置。此后所有对话请求都先经过 LLM Manager 路由。

> ⚠️ **关键**：LLM Manager 实例的 `model` 字段必须是占位符（如 `main`），不能是你目录里的真实模型名。否则 AstrBot 每次请求都会固定用那个模型，`/llm use` 切换不会生效。

---

## 快速上手

```
/llm import          # 第一步：从系统配置导入所有供应商
/llm list            # 第二步：查看所有模型（带序号）
/llm use 3           # 第三步：切换到序号 3 的模型
/llm test            # 测试当前模型连通性
```

---

## 指令详解

> 所有 `/llm` 指令默认仅管理员可用。如需放开，在 WebUI 插件配置中关闭 `admin_only`。

---

### 📋 `/llm list` — 查看所有模型

按三级结构展示：API Base URL → 实例 → 模型。每个模型前有全局序号，切换时用序号即可。

**参数：**

| 参数 | 说明 |
|---|---|
| （无） | 默认渲染成卡片图片 |
| `-t` | 纯文本模式（不发图片） |
| `关键词` | 过滤包含关键词的行（仅纯文本模式下有效） |

**例子：**

```
/llm list                  # 卡片图片查看
/llm list -t               # 纯文本查看
/llm list -t deepseek      # 纯文本过滤，只看含 deepseek 的行
/llm list -t gemini        # 纯文本过滤，只看含 gemini 的行
```

图片卡片顶部会显示**当前正在使用的模型和序号**，每个模型带序号徽章和能力标签（工具/视觉/语音），停用的模型会标注。

---

### 🔄 `/llm use` — 切换模型

切换当前使用的模型。支持按序号、模型 ID、实例 ID 切换。

**参数：**

| 参数 | 说明 |
|---|---|
| `<序号>` | 按 `/llm list` 里的序号切换（最常用） |
| `<模型id>` | 按模型名称切换，如 `deepseek-chat` |
| `<实例id>` | 按实例 ID 切换，切到该实例的默认模型 |
| `-c` | 仅当前会话切换（不影响全局和其他会话） |
| `global` | 清除当前会话的独立切换，回到全局默认 |
| （无参数） | 查看当前生效的模型和切换状态 |

**例子：**

```
/llm use 3                  # 全局切换到序号 3 的模型
/llm use deepseek-reasoner  # 全局切换到模型 deepseek-reasoner
/llm use deepseek_推理      # 全局切换到实例 deepseek_推理 的默认模型
/llm use 4 -c               # 仅当前会话切换到序号 4（其他会话不受影响）
/llm use global             # 清除当前会话切换，回到全局默认
/llm use                    # 查看当前生效的模型、全局默认、会话切换状态
```

---

### ⭐ `/llm default` — 设置全局默认模型

设置全局默认模型。所有没有独立会话切换的对话都会用这个模型。

**参数：**

| 参数 | 说明 |
|---|---|
| `<序号>` | 按序号设为全局默认 |
| `<模型id>` | 按模型名称设为全局默认 |
| `<实例id>` | 按实例 ID 设为全局默认（用该实例的默认模型） |
| （无参数） | 查看当前全局默认 |

**例子：**

```
/llm default 3              # 把序号 3 的模型设为全局默认
/llm default                # 查看当前全局默认
```

> `/llm use` 和 `/llm default` 的区别：`/llm use` 是临时切换（重启后可能丢失），`/llm default` 是持久化的全局默认。日常切换用 `/llm use` 即可。

---

### 📥 `/llm import` — 从系统配置导入

从 AstrBot 的 `cmd_config.json` 导入已有的供应商和模型。这是安装后的第一步。

**参数：** 无

**例子：**

```
/llm import
```

导入时会：
- 自动读取所有 `provider_sources`（API Base URL + key）和模型
- 按三级结构导入插件管理
- **自动读取你原来的默认模型并设为插件全局默认**，确保安装插件后默认模型不变
- dashscope 等 agent_runner 类型会自动跳过

> ⚠️ 必须在你把 WebUI 的 default_provider_id 改成 LLM Manager **之前**执行，否则原来的默认模型值已被覆盖，无法自动保留。

---

### 🔄 `/llm resync` — 从系统配置完全重建

如果你**手动在 AstrBot WebUI 删除或修改了供应商**，插件的配置不会自动同步，`/llm list` 还会显示旧数据。此时用这个指令完全重建。

**参数：** 无

**例子：**

```
/llm resync
```

重建时会：
- 先清空插件现有配置，再从 `cmd_config.json` 重新导入
- 保留你设置的全局默认模型（如果目标仍存在）
- 清理无效的会话切换

> 与 `/llm import` 的区别：`import` 是增量导入（已有的不重复添加），`resync` 是完全重建（先清空再导入）。手动删除供应商后用 `resync`。

---

### ➕ `/llm add` — 手动添加后端（一级源）

添加一个新的 API Base URL 和 key。适用于不在系统配置里的新供应商。

**参数：**

| 参数 | 说明 |
|---|---|
| `<站名>` | 给这个后端起个名字，如 `DeepSeek`、`硅基流动` |
| `<base_url>` | API 地址，如 `https://api.deepseek.com/v1` |
| `<key>` | API Key |
| `[类型]` | 可选，提供商类型，默认 `openai_chat_completion` |

**例子：**

```
/llm add DeepSeek https://api.deepseek.com/v1 sk-xxxx
/llm add 硅基流动 https://api.siliconflow.cn/v1 sk-yyyy
/llm add Claude https://api.anthropic.com sk-zzzz openai_chat_completion
```

添加后需要用 `/llm group add` 建分组，再用 `/llm model add` 挂模型。

---

### 👥 `/llm group` — 分组实例管理

管理二级实例（分组）。每个实例属于一个一级源，实例 ID 自动生成为 `站名_分组名`。

**子命令：**

| 子命令 | 说明 |
|---|---|
| `add` | 新增分组实例 |
| `rm` | 删除分组实例（含其下所有模型） |
| `enable` | 启用分组实例 |
| `disable` | 停用分组实例（路由时自动跳过） |

**`/llm group add` 参数：**

| 参数 | 说明 |
|---|---|
| `<站名>` | 所属一级源的站名 |
| `<分组名>` | 分组名称，如 `通用`、`推理`、`高速` |
| `[模型...]` | 可选，创建时直接挂载的模型 ID |

**例子：**

```
/llm group add DeepSeek 通用 deepseek-chat deepseek-v3     # 建分组并挂模型
/llm group add DeepSeek 推理                                 # 建空分组
/llm group rm deepseek_推理                                  # 删除实例
/llm group enable deepseek_通用                              # 启用实例
/llm group disable deepseek_通用                             # 停用实例
```

---

### 🧠 `/llm model` — 模型挂载管理

管理三级模型。在实例上挂载或移除模型。

**子命令：**

| 子命令 | 说明 |
|---|---|
| `add` | 给实例挂载模型 |
| `rm` | 从实例移除模型 |

**参数：**

| 参数 | 说明 |
|---|---|
| `<实例id>` | 实例 ID，如 `deepseek_通用` |
| `<模型id>...` | 一个或多个模型 ID，如 `deepseek-chat` |

**例子：**

```
/llm model add deepseek_通用 deepseek-chat deepseek-v3      # 挂载多个模型
/llm model add deepseek_推理 deepseek-r1-0528               # 挂载单个模型
/llm model rm deepseek_通用 deepseek-v3                      # 移除模型
```

---

### 🗑️ `/llm rm` — 统一删除

删除一级源、二级实例或三级模型。

**子命令：**

| 子命令 | 说明 |
|---|---|
| `source` | 删除一级源（含其下所有分组和模型） |
| `instance` | 删除二级实例（含其下所有模型） |
| `model` | 删除三级模型 |

**例子：**

```
/llm rm source DeepSeek              # 删除一级源 DeepSeek（含所有分组和模型）
/llm rm instance deepseek_推理       # 删除实例 deepseek_推理
/llm rm model deepseek_通用 deepseek-v3   # 从实例 deepseek_通用 移除模型 deepseek-v3
/llm rm                              # 查看删除用法
```

> 删除后立即生效，用 `/llm list` 查看最新配置。

---

### ✅ `/llm enable` / `/llm disable` — 启停实例

快速启用或停用一个实例。停用的实例在路由时会被自动跳过。

**参数：**

| 参数 | 说明 |
|---|---|
| `<实例id>` | 实例 ID，如 `deepseek_通用` |

**例子：**

```
/llm enable deepseek_通用       # 启用实例
/llm disable deepseek_通用      # 停用实例（路由自动跳过）
```

> 也可以用 `/llm group enable` 和 `/llm group disable`，效果相同。

---

### 🧪 `/llm test` — 连通性与时延测试

测试模型的连通性和响应延迟。支持批量测试多个模型。

**参数：**

| 参数 | 说明 |
|---|---|
| （无） | 测试**当前正在使用**的模型 |
| `<序号>...` | 测试一个或多个指定序号的模型（空格分隔，最多 10 个） |
| `<模型id>...` | 按模型 ID 测试 |
| `<实例id>...` | 按实例 ID 测试（测该实例的默认模型） |

**例子：**

```
/llm test                  # 测试当前正在使用的模型
/llm test 3                # 测试序号 3 的模型
/llm test 3 5 7            # 批量测试序号 3、5、7 的模型（并行）
/llm test 30 31 36         # 批量测试多个模型
/llm test deepseek-chat    # 按模型 ID 测试
```

批量测试时会并行发起请求，结果按序号排列显示，每行标注 ✅/❌ + 延迟或错误信息，末尾显示成功数/总数。

---

### ❓ `/llm help` — 查看帮助

查看所有指令的简要说明。

**参数：** 无

**例子：**

```
/llm help
```

---

## 指令总表

| 指令 | 说明 | 常用度 |
|---|---|---|
| `/llm list [-t] [关键词]` | 查看所有模型（三级结构，带序号） | ⭐⭐⭐ |
| `/llm use <目标> [-c]` | 切换模型（序号/模型ID/实例ID） | ⭐⭐⭐ |
| `/llm default <目标>` | 设置全局默认模型 | ⭐⭐ |
| `/llm import` | 从系统配置导入供应商 | ⭐⭐⭐ |
| `/llm resync` | 从系统配置完全重建 | ⭐⭐ |
| `/llm add <站名> <url> <key>` | 添加一级源 | ⭐ |
| `/llm group add <站名> <分组名> [模型...]` | 建分组实例 | ⭐ |
| `/llm group rm/enable/disable <实例id>` | 管理实例 | ⭐ |
| `/llm model add/rm <实例id> <模型id>...` | 挂载/移除模型 | ⭐ |
| `/llm rm source/instance/model <目标>` | 统一删除 | ⭐ |
| `/llm enable/disable <实例id>` | 启停实例 | ⭐ |
| `/llm test [目标...]` | 连通性与时延测试（支持批量） | ⭐⭐ |
| `/llm help` | 查看帮助 | ⭐ |

---

## 路由规则（优先级从高到低）

1. **请求显式指定的模型**（命中目录时）
2. **会话级切换**（`/llm use -c`，按 unified_msg_origin 区分会话）
3. **全局默认**（`/llm default`）
4. **第一个启用的模型**（兜底）

**能力适配**：路由时若目标模型不支持图片 / 音频 / 工具调用，会自动降级（忽略对应输入并记录日志），不会把不支持的能力硬塞给模型。

---

## 常见问题

**Q: 切换模型后聊天还是用原来的模型？**
A: 检查 WebUI 中 LLM Manager 实例的 `model` 字段是否为占位符（如 `main`），不能填真实模型名。同时确认 `default_provider_id` 已设为 LLM Manager 实例。

**Q: 手动在 WebUI 删除了供应商，`/llm list` 还显示？**
A: 插件配置和 AstrBot 配置是独立的。执行 `/llm resync` 从系统配置完全重建即可。

**Q: 想让群成员也能用切换指令？**
A: 在 WebUI 插件配置中关闭 `admin_only`，所有 `/llm` 指令就对所有人开放了。

**Q: 安装插件后默认模型变了？**
A: 确保先执行 `/llm import`（此时原默认模型还在），再改 WebUI 的 default_provider_id。导入时会自动保留原默认模型。

---

## 数据与配置

- **配置仓库**：`data/plugin_data/astrbot_plugin_llm_manager/backends.json`（三级结构 + 默认与会话切换）
- **插件配置**：WebUI 插件配置中 `admin_only`（默认开启）控制所有 `/llm` 指令是否仅管理员可用
- 所有 API Key 只存在插件自己的 backends.json 中，不写入 AstrBot 配置

---

## 兼容性

- 面向 AstrBot 4.23+（在 4.26.x 上开发验证）
- 委托层复用 AstrBot 内置 Provider（OpenAI 兼容 / Anthropic / Gemini / 小米等），新增类型无需改本插件
- 运行时自动检测并安装 Pillow（用于 `/llm list` 图片渲染）
