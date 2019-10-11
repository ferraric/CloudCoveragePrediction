import datetime


def add_days(year, month, day, n):
    d = datetime.datetime(year, month, day)
    DD = datetime.timedelta(days=n)
    new_date = d + DD
    return (new_date.year, new_date.month, new_date.day)


def subtract_days(year, month, day, n):
    d = datetime.datetime(year, month, day)
    DD = datetime.timedelta(days=n)
    new_date = d - DD
    return (new_date.year, new_date.month, new_date.day)
