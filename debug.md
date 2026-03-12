# GIDS 异步读取排障记录

## 背景

本轮排障的目标是把 `GIDS/BaM` 的特征读取路径改成 `submit + wait` 的 split-phase 形式，并确认：

1. `submit_async / wait_async` 这条链路是否正确返回数据
2. 当前实现是否真的具备异步 overlap

主要涉及文件：

- `bam/include/page_cache.h`
- `gids_module/gids_kernel.cu`
- `gids_module/gids_nvme.cu`
- `evaluation/run_homogenous_train.sh`
- `evaluation/homogenous_train.py`

相关提交：

- `2e2e26f` `fix(gids): 修复异步读取链路并补充同步基线`
- `18656b7` `refactor(gids_module): 移除未使用的 IOStack 与 agile host`

## 初始现象

训练最开始可以跑通 1 个 iter，但读取出的特征数据不对，`Test Acc` 长时间停在 `23.03%`。

最初的怀疑点有两个：

1. `submit` 阶段 `find_slot()` 分配的槽位不对
2. `wait` 阶段没有正确接收到 `submit` 写入的槽位

同时，早期日志里 `wait:` 输出很少，设备端打印的地址和 `idx_idx` 也明显异常。

## 排查过程

### 1. 先确认当前代码实际跑的是哪条路径

一开始 `read_feature()` 不是直接跑异步链，而是先执行一次同步 `read_feature_kernel`，随后才执行 `submit_async` 和 `wait_async`。

这导致：

- cache 被同步路径提前热起来
- 后面的 `submit/wait` 大量命中
- `wait` 日志极少，无法真实暴露 miss 路径问题

因此第一步先去掉了 `read_feature()` 里对同步 `read_feature_kernel` 的预热调用，只保留 `submit + wait`。

### 2. 给 host 侧加上下文观测

为了避免继续被设备端 `%p` 和异常 `idx_idx` 打印误导，在 `gids_nvme.cu` 里增加了 host 侧 `dump_warp_ctxs()`，打印前 8 个 warp 的：

- `idx_idx`
- `row_index`
- `isHit`
- `page_trans`
- `cid`
- `ctrl`
- `queue`
- `base_master`

这样可以直接判断：

- `submit` 是否写好了上下文
- `wait` 是否读到了相同的上下文

### 3. 定位到 wait kernel 只补了前 32 维

去掉同步预热后，日志显示：

- `submit` 数量正常
- `wait` 数量只有极少数
- host 侧 `ctx` 是自洽的

进一步看 `read_feature_kernel_wait_async()` 后发现，原来的 `if (!ctx.isHit)` 写在 `for (tid += 32)` 循环内部。

结果是：

1. 第一次 successful wait 之后，`ctx.isHit` 被置为 `true`
2. 同一个 warp 后续维度循环被跳过
3. 一整行只有前 32 维被正确覆盖，后续维度保留了 submit 阶段写入的占位 `0`

修复方式：

- 把 `if (!ctx.isHit)` 提到循环外
- miss warp 一旦进入 wait，就把整行特征都补完

修复后，`wait` 数量从个位数恢复到正常规模。

### 4. 修 `page_trans` 与 `base_master` 的交接

在 `page_cache.h` 里，异步 miss 路径原先存在几处上下文交接不完整的问题：

1. `submit` 侧没有完整写回 `ctx.eq_mask / ctx.master / ctx.count / ctx.base_master`
2. `wait` 侧仍依赖 `ctx.base_master` 构造返回地址
3. `count` 在 wait 路径里可能是未初始化局部变量

修复后：

- `submit` 会把上述字段完整写回 `ctx`
- `wait` 会使用本轮有效的 `base_master`
- `ctx.base_master` 在 wait 结束后回写，避免返回地址漂移

### 5. 修 `NV_B` 对槽位发布的竞争窗口

在 `NV_NB -> NV_B` 的 miss 交接里，原先 follower warp 会直接读 `pages[index].offset`，并把 `0` 近似当作“尚未发布”。

这个做法不可靠，因为：

- `0` 可能是合法槽位
- leader 线程可能刚设置 `BUSY`，但还没来得及写完真实 `page_trans`

修复方式：

1. 引入 `ASYNC_PAGE_TRANS_PENDING = 0xFFFFFFFFU`
2. leader 在分配槽位前先写 pending sentinel
3. follower 在 `NV_B` 分支里等待：
   - `offset != pending`
   - `cache_page.page_translation` 与当前逻辑页匹配

这样可以避免 follower 过早消费脏槽位。

### 6. 加同步基线对照，排除“异步特有 bug”

为了回答“是不是异步读错了、同步本来就是对的”这个问题，在 `gids_nvme.cu` 里加入了 `GIDS_FORCE_SYNC_READ` 开关。

当环境变量开启时：

- `read_feature()` 直接走 `read_feature_kernel`
- 不走 `submit_async / wait_async`

同时修改了 `evaluation/run_homogenous_train.sh`，用 `sudo env` 把 `GIDS_FORCE_SYNC_READ` 透传进去。

这样就可以直接比较：

- 默认模式：异步 split-phase
- `GIDS_FORCE_SYNC_READ=1`：原始同步读取

### 7. 做异步结果对同步结果的逐元素比较

为了更快收敛，在 `read_feature()` 里又增加了一段调试逻辑：

- wait 完成后
- 用同步 `read_feature_kernel` 重新读取前 2 行
- 比较前 2 行、全部 `dim` 维度的逐元素结果

调试输出格式：

```text
async debug: rows=2 dims=8 mismatches=0/2048
```

其中：

- `rows=2` 表示比较前两行
- `2048 = 2 * 1024`，是全量维度比较，不只是打印的前 8 维

## 最终定位结果

### 已确认修复的问题

1. `read_feature()` 先同步预热 cache，导致异步 miss 路径被遮蔽
2. `wait` kernel 只补前 32 维，导致整行大部分特征仍为 0
3. `submit -> wait` 的 `ctx` 交接不完整，`base_master/count` 可能错误
4. `NV_B` 通过 `offset == 0` 近似判断发布完成，存在竞争窗口

### 已确认的正确性结论

从 `tmp4` 阶段的日志看：

- host 侧 `submit ctx sample` 和 `wait ctx sample` 一致
- `async debug: mismatches=0/2048`

这说明：

```text
异步 submit/wait 结果 == 同步 read_feature_kernel 结果
```

至少在抽样验证的前 2 行上，异步链路已经能返回与同步 page-cache 读取一致的数据。

### 仍然成立的限制

虽然当前代码已经实现了 split-phase 读取，但还不能说“真正异步”。

原因是 `read_feature()` 里仍然有明显的全局同步点：

- `submit` 后立即 `cudaDeviceSynchronize()`
- `wait` 后立即 `cudaDeviceSynchronize()`
- 入口前还有一次同步

所以现在的状态更准确地说是：

```text
逻辑上分成了 submit + wait
但执行上仍然基本串行，没有形成真正的 overlap
```

也就是说：

- 现在可以验证异步链路的正确性
- 但还不能宣称已经实现了真正的异步 IO 流水

## 当前代码状态

### `page_cache.h`

已完成：

- `ASYNC_PAGE_TRANS_PENDING` 发布协议
- `ctx.eq_mask/master/count/base_master` 回写
- `wait` 使用有效 `base_master` 取返回页地址
- `submit/wait/normal` 调试打印限流

### `gids_kernel.cu`

已完成：

- `read_feature_kernel_wait_async()` 改为整行补写

### `gids_nvme.cu`

已完成：

- 去掉 `read_feature()` 中原有的同步预热
- 增加 host 侧 `ctx` dump
- 增加 `async vs sync` 对照
- 增加 `GIDS_FORCE_SYNC_READ` 同步基线

### `evaluation`

已完成：

- `run_homogenous_train.sh` 支持透传 `GIDS_FORCE_SYNC_READ`
- `homogenous_train.py` 暂时限制到 20 iter，便于调试和收集日志

## 如何复现验证

### 1. 跑异步 split-phase 路径

```bash
bash /home/wzq/GIDS_IO/evaluation/run_homogenous_train.sh
```

关注输出：

- `submit ctx sample`
- `wait ctx sample`
- `async debug: rows=2 dims=8 mismatches=...`

### 2. 跑同步基线

```bash
GIDS_FORCE_SYNC_READ=1 bash /home/wzq/GIDS_IO/evaluation/run_homogenous_train.sh
```

关注输出：

- `read_feature mode: sync baseline via read_feature_kernel`

### 3. 判断“是否真正异步”

当前只能证明“走了 submit/wait 路径”，不能证明“有 overlap”。

要验证真正异步，需要进一步做 timeline 级别观测，例如：

- `nsys profile`
- `cudaEvent` 打点
- `NVTX` 区间标记

如果时间线仍然是：

```text
submit_async -> sync -> wait_async -> sync
```

那就说明现在仍然是串行 split-phase，而不是真正的异步流水。

## 当前结论

截至本轮排障结束，可以明确确认：

1. 异步 `submit/wait` 数据接收链路已经基本打通
2. 之前导致数据错误的主要 bug 已修复
3. 当前异步结果与同步 `read_feature_kernel` 在抽样对照上是一致的
4. 但代码里仍有同步屏障，所以还不能认为已经实现了真正的异步 overlap

后续如果继续推进，重点不该再放在 `page_trans` 正确性，而应该转向：

1. 用 timeline 工具验证是否存在真实 overlap
2. 去掉 `read_feature()` 中阻断流水的 `cudaDeviceSynchronize()`
3. 把 split-phase 真正改成 event/stream 依赖的异步执行模型
