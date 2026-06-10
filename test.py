from pytdx.hq import TdxHq_API

# 直接用已知能通的服务器
api = TdxHq_API()
api.connect('218.75.126.9', 7709)
data = api.get_security_bars(2, 1, '688981', 0, 3)
print(data)
api.disconnect()

# 测试 connector
from src.tdx.connector import TdxConnector, LiveBarBuffer

with TdxConnector() as conn:
    buf = LiveBarBuffer(conn, symbol="sh688981", capacity=32)
    ok = buf.update()
    print("更新成功:", ok)
    print(buf.get_df().tail(3))