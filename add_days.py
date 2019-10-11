def add_days(year, month, day, n):
    if (month == "03" or month == "05" or month == "01" or month == "07"
            or month == "08" or month == "10" or month == "12"):
        k = 31 - n
        if (int(day) <= k):
            actual = int(day) + n
            new_month = month
        else:
            p1 = 31 - int(day)
            p2 = n - p1
            if p2 < 10:
                actual = "0" + str(p2)
            else:
                actual = str(p2)
            if (int(month) != 12):
                new_month = int(month) + 1
            else:
                new_month = 1
                year = int(year) + 1
            if int(new_month) < 10:
                new_month = "0" + str(new_month)
            else:
                new_month = new_month
    elif (month == "04" or month == "06" or month == "09" or month == "11"):
        k = 30 - n
        if (int(day) <= k):
            actual = int(day) + n
            new_month = month
        else:
            p1 = 30 - int(day)
            p2 = n - p1
            if p2 < 10:
                actual = "0" + str(p2)
            else:
                actual = str(p2)
            if (int(month) != 12):
                new_month = int(month) + 1
            else:
                new_month = 1
                year = int(year) + 1
            if int(new_month) < 10:
                new_month = "0" + str(new_month)
            else:
                new_month = str(new_month)
    else:
        if (year != "2016"):
            k = 28 - n
            if (int(day) <= k):
                actual = int(day) + n
                new_month = month
            else:
                p1 = 28 - int(day)
                p2 = n - p1
                if p2 < 10:
                    actual = "0" + str(p2)
                else:
                    actual = str(p2)
                if (int(month) != 12):
                    new_month = int(month) + 1
                else:
                    new_month = 1
                    year = int(year) + 1
                if int(new_month) < 10:
                    new_month = "0" + str(new_month)
                else:
                    new_month = new_month
        else:
            k = 29 - n
            if (int(day) <= k):
                actual = int(day) + n
                new_month = month
            else:
                p1 = 29 - int(day)
                p2 = n - p1
                if p2 < 10:
                    actual = "0" + str(p2)
                else:
                    actual = str(p2)
                if (int(month) != 12):
                    new_month = int(month) + 1
                else:
                    new_month = 1
                    year = int(year) + 1
                if int(new_month) < 10:
                    new_month = "0" + str(new_month)
                else:
                    new_month = new_month
    return (str(year), str(new_month), str(actual))
