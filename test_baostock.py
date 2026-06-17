import baostock as bs
bs.login()
rs = bs.query_history_k_data_plus(
    "sh.688981", "date,time,open,high,low,close,volume,amount",
    start_date="2025-01-01", end_date="2026-06-12", frequency="30"
)
df = rs.get_data()
print(df.tail())   # 看最新一行的日期，确认数据覆盖到哪天
bs.logout()