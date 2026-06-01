# 用 SSH 上传项目到 GitHub

以下步骤适用于把本地代码上传到 GitHub，示例仓库为：

```text
git@github.com:xujiayuxu/CBNI-NCSAM.git
```

## 1. 生成 SSH Key

```bash
ssh-keygen -t ed25519 -C "你的GitHub邮箱"
```

如果提示保存路径，直接回车即可。生成后查看公钥：

```bash
cat ~/.ssh/id_ed25519.pub
```

如果你使用了自定义文件名，例如 `id_ed25519_github`，则执行：

```bash
cat ~/.ssh/id_ed25519_github.pub
```

## 2. 添加公钥到 GitHub

打开：

```text
https://github.com/settings/keys
```

点击 `New SSH key`：

```text
Title: 自定义名称，例如 My Linux PC
Key type: Authentication Key
Key: 粘贴 cat 命令输出的整行公钥
```

保存即可。

## 3. 测试 SSH 是否成功

```bash
ssh -T git@github.com
```

第一次连接如果提示：

```text
Are you sure you want to continue connecting?
```

输入：

```text
yes
```

看到类似下面的提示就表示成功：

```text
Hi 用户名! You've successfully authenticated, but GitHub does not provide shell access.
```

## 4. 初始化并提交本地项目

进入项目目录：

```bash
cd /home/xjy/code/CBN_NCSAM
```

如果还没有初始化 Git：

```bash
git init
git branch -M main
```

添加文件并提交：

```bash
git add .
git commit -m "feat: initial project upload"
```

## 5. 设置 GitHub SSH 远程地址

```bash
git remote add origin git@github.com:xujiayuxu/CBNI-NCSAM.git
```

如果已经有 `origin`，改用：

```bash
git remote set-url origin git@github.com:xujiayuxu/CBNI-NCSAM.git
```

检查远程地址：

```bash
git remote -v
```

## 6. 推送到 GitHub

```bash
git push -u origin main
```

之后再提交新代码时，只需要：

```bash
git add .
git commit -m "你的提交说明"
git push
```

## 常见问题

如果提示 `Permission denied (publickey)`，说明 GitHub 没有识别你的 SSH key。检查：

```bash
ssh -T git@github.com
```

如果你使用的是自定义 key 文件，例如 `~/.ssh/id_ed25519_github`，可以给当前仓库指定 SSH key：

```bash
git config core.sshCommand "ssh -i ~/.ssh/id_ed25519_github -o IdentitiesOnly=yes"
```

然后再推送：

```bash
git push -u origin main
```
