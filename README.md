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

> 现有供应商的 base_url / key 可在 `data/cmd_config.json` 的 `provider_sources` 里查到，用 `/llm add` 逐条导入即可（key 不需要重复填多次）。

## 使用

### 0. 从系统配置一键导入（推荐）

如果你已经在 AstrBot WebUI 里配过供应商，直接：

```
/llm import
```

会自动读取 `cmd_config.json` 里全部 `provider_sources` 和模型，按三级结构导入插件管理（dashscope 等 agent_runner 类型会自动跳过）。导入时还会**自动读取你原来的默认模型并设为插件全局默认**，确保安装插件后默认模型不变。导入后用 `/llm list` 查看。

> 注意：导入必须在你把 default_provider_id 改成 LLM Manager 之前执行，否则原来的默认模型值已被覆盖，无法自动保留。

### 0.1 手动修改后同步（重要）

如果你**手动在 AstrBot WebUI 删除或修改了供应商**，插件的 `backends.json` 不会自动同步，`/llm list` 还会显示旧数据。此时执行：

```
/llm resync
```

会从 `cmd_config.json` **完全重建**插件配置（先清空再导入），同时保留你设置的全局默认模型（如果目标仍存在），清理无效的会话切换。重建后用 `/llm list` 查看最新配置。

### 1. 手动添加后端（一级源）

```
/llm add DeepSeek https://api.deepseek.com/v1 sk-xxxx
/llm add 硅基流动 https://api.siliconflow.cn/v1 sk-yyyy
/llm add Claude https://api.anthropic.com openai_chat_completion   # 类型可显式指定
```

### 2. 建分组（二级实例，id 自动生成 `站名_分组名`）

```
/llm group add DeepSeek 通用 deepseek-chat deepseek-v3
/llm group add DeepSeek 推理 deepseek-reasoner
/llm model add deepseek_推理 deepseek-r1-0528        # 后续继续挂模型
```

### 3. 查看（三级卡片图片）

```
/llm list            # 默认渲染成卡片图片
/llm list -t         # 纯文本模式
/llm list deepseek   # 关键词过滤（纯文本模式下）
```

图片示例：顶部标题栏显示源/模型总数，每个源一张卡片，实例缩进显示，模型带序号徽章和能力标签（工具/视觉/语音），当前默认模型高亮标注。

### 4. 切换模型

```
/llm use 3                  # 全局切换（按序号）
/llm use deepseek-reasoner  # 按模型 ID
/llm use deepseek_推理      # 按实例 ID（切到该组默认模型）
/llm use 4 -c               # 仅当前会话切换
/llm use global             # 清除会话切换，回到全局默认
/llm default 3              # 设置全局默认
```

### 5. 运维

```
/llm test 3                 # 连通性与时延
/llm group rm deepseek_推理 # 删除实例
/llm disable deepseek_通用  # 停用实例（路由自动跳过）
/llm help                   # 全部指令
```

## 指令总表

> 所有 `/llm` 指令默认仅管理员可用。

| 指令 | 说明 | 权限 |
|---|---|---|
| `/llm list [关键词]` | 三级树形查看，模型带全局序号（默认渲染图片，`-t` 纯文本） | 管理员 |
| `/llm use <序号\|模型id\|实例id> [-c]` | 全局 / 会话切换；不带参数查看当前生效 | 管理员 |
| `/llm default <目标>` | 设置全局默认 | 管理员 |
| `/llm import` | 从系统配置(cmd_config.json)导入已有供应商，自动保留原默认模型 | 管理员 |
| `/llm resync` | 从系统配置完全重建（解决手动删除供应商后不同步的问题） | 管理员 |
| `/llm add <站名> <base_url> <key> [类型]` | 新增一级源 | 管理员 |
| `/llm group add <站名> <分组名> [模型...]` | 新增分组实例 | 管理员 |
| `/llm group rm \| enable \| disable <实例id>` | 管理实例 | 管理员 |
| `/llm model add \| rm <实例id> <模型id>...` | 挂载/移除模型 | 管理员 |
| `/llm rm source \| instance \| model <目标>` | 统一删除（源/实例/模型） | 管理员 |
| `/llm enable \| disable <实例id>` | 启停实例 | 管理员 |
| `/llm test <目标>` | 连通性与时延 | 管理员 |
| `/llm help` | 查看指令帮助 | 管理员 |

## 路由规则（优先级从高到低）

1. 请求显式指定的模型（命中目录时）
2. 会话级切换（`-c`，按 unified_msg_origin）
3. 全局默认（`/llm default`）
4. 第一个启用的模型

能力适配：路由时若目标模型不支持图片 / 音频 / 工具调用，会自动降级（忽略对应输入并记录日志），不会把不支持的能力硬塞给模型。

## 数据与配置

- 配置仓库：`data/plugin_data/astrbot_plugin_llm_manager/backends.json`（三级结构 + 默认与会话切换）。
- 插件配置：WebUI 插件配置中 `admin_only`（默认开启）控制所有 `/llm` 指令是否仅管理员可用；关闭后群内所有成员均可使用。
- 所有 API Key 只存在插件自己的 backends.json 中，不写入 AstrBot 配置。

## 兼容性

- 面向 AstrBot 4.23+（在 4.26.x 上开发验证）。低于该版本若 `astrbot.core.provider.register` 等内部路径不存在，插件会在加载时报错，不会影响 AstrBot 本体。
- 委托层复用 AstrBot 内置 Provider（OpenAI 兼容 / Anthropic / Gemini / 小米等），新增类型无需改本插件。
