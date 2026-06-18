from datetime import datetime, timedelta


def iter_days(from_date: str, to_date: str):
    start = datetime.strptime(from_date, "%Y%m%d")
    end = datetime.strptime(to_date, "%Y%m%d")

    current = start
    while current <= end:
        day = current.strftime("%Y%m%d")
        yield day
        current += timedelta(days=1)