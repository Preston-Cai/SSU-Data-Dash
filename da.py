import os
os.environ["DATA_DIR"] = r"C:\Users\jiang\.cache\huggingface\hub\datasets--Baliyang--SSU_data_dash\snapshots\96a258439e2a85f650699be8339e52d8ac5614b7"   # only if you downloaded the files yourself

import data


listings = data.load("listings")
print(
    f"{len(listings):,} sales, {listings['date_sold'].min():%B %Y} to {listings['date_sold'].max():%B %Y}")
listings.head()

off_market = data.load("off_market")
list_aware = data.load("list_aware")
print(f"{len(off_market):,} homes to value with no list price (off-market)")
print(f"{len(list_aware):,} homes to value that have a list price (list-aware)")
off_market.head()

# A first look: the median sold price each month, by property type.
monthly = listings.groupby([listings["date_sold"].dt.to_period("M"), "house_type_name"])[
    "sold_price"].median()
monthly.unstack().plot(figsize=(9, 4), ylabel="Median sold price ($)", xlabel="");
