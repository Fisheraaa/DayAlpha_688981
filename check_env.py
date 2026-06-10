"""
check_env.py — 运行环境一键检测
运行方式：python check_env.py
"""
import sys

print("=" * 52)
print("  688981 T+0 策略 · 环境检测")
print("=" * 52)

# Python版本
print(f"\n🐍 Python: {sys.version.split()[0]}")

# PyTorch & CUDA
try:
    import torch
    print(f"🔦 PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        idx  = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem  = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        cc_major = torch.cuda.get_device_properties(idx).major
        cc_minor = torch.cuda.get_device_properties(idx).minor
        print(f"✅ CUDA 可用: {torch.version.cuda}")
        print(f"   GPU: {name}")
        print(f"   显存: {mem:.1f} GB")
        print(f"   Compute Capability: {cc_major}.{cc_minor}")

        # Blackwell (sm_120) 需要 PyTorch nightly
        if cc_major >= 12:
            tv = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
            # nightly版本号格式不同，用try判断实际能否跑CUDA
            try:
                t = torch.tensor([1.0]).cuda()
                _ = t * 2
                print(f"   架构: Blackwell (sm_{cc_major}{cc_minor}) ✅ 当前PyTorch可正常使用")
            except Exception:
                print(f"   架构: Blackwell (sm_{cc_major}{cc_minor}) ⚠️  当前PyTorch不支持！")
                print(f"   请执行：")
                print(f"   pip uninstall torch torchvision torchaudio -y")
                print(f"   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128")
        else:
            pass  # cc_major < 12，已在上面打印过CC了
        amp_ok = cc_major >= 7
        print(f"   AMP Tensor Core: {'✅ 支持' if amp_ok else '⚠️ 不支持（CC<7.0）'}")
        compile_ok = hasattr(torch, "compile")
        print(f"   torch.compile: {'✅ 可用' if compile_ok else '⚠️ 不可用（PyTorch<2.0）'}")

        # 建议batch_size
        # RTX 5060/4060 等8GB卡用256，10GB+用512
        bs = 512 if mem >= 10 else (256 if mem >= 6 else 128)
        print(f"   建议 batch_size: {bs}  (8GB显存推荐256，避免OOM)")
    else:
        print("⚠️  CUDA 不可用 — 将使用CPU（较慢）")
        print("   如需启用GPU，请执行：")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
except ImportError:
    print("❌ PyTorch 未安装: pip install torch")

# 其他依赖
deps = [
    ("pandas",      "pandas"),
    ("numpy",       "numpy"),
    ("sklearn",     "scikit-learn"),
    ("baostock",    "baostock"),
    ("optuna",      "optuna"),
    ("shap",        "shap"),
    ("pytdx",       "pytdx（通达信，可选）"),
    ("yaml",        "pyyaml"),
]
print("\n📦 依赖包:")
for mod, label in deps:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "OK")
        print(f"  ✅ {label}: {ver}")
    except ImportError:
        mark = "⚠️ " if "可选" in label else "❌"
        print(f"  {mark} {label}: 未安装")

print("\n" + "=" * 52)
print("  若所有 ✅ 均已显示，可直接运行：")
print("  python main.py --mode full")
print("=" * 52 + "\n")
