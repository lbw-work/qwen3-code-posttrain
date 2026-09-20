"""仅在禁网 Docker 容器中运行。标准输入/输出都是 JSON。宿主机不要直接执行它。"""

import json
import subprocess
import sys


CHILD = r'''
import contextlib, json, os, signal, sys
payload = json.load(sys.stdin)
code = payload["code"]
tests = payload["tests"]
passed = 0
with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
    for test in tests:
        # 每条测试重新执行候选代码，避免上一条测试改了全局变量影响下一条。
        namespace = {"__name__": "__main__"}
        try:
            signal.alarm(1)
            exec(code, namespace)
            exec(test, namespace)
            passed += 1
        except BaseException:
            pass
        finally:
            signal.alarm(0)
print(json.dumps({"passed": passed, "total": len(tests)}))
'''


def main() -> None:
    samples = json.load(sys.stdin)
    scores = []
    for sample in samples:
        tests = sample["tests"]
        if not isinstance(tests, list) or not tests or any(not isinstance(x, str) for x in tests):
            scores.append(0.0)
            continue
        try:
            # 候选代码在单独子进程执行。卡死、退出或无合法结果一律得 0。
            process = subprocess.run(
                [sys.executable, "-I", "-c", CHILD], input=json.dumps(sample),
                text=True, capture_output=True, timeout=12,
            )
            result = json.loads(process.stdout)
            scores.append(result["passed"] / result["total"] if process.returncode == 0 else 0.0)
        except (subprocess.TimeoutExpired, ValueError, KeyError, ZeroDivisionError):
            scores.append(0.0)
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
