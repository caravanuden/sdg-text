"""
This file queries for GDELT articles relating to the articles we're given
"""
from google.cloud import bigquery
import csv


LAT_AND_LONG_SQUARE_LENGTH = 4

def get_query_string(lat, long, year):
    """

    :param lat (float): lat correspnding to the given row
    :param long (float):
    :param year (int):
    :return: (string) the query corresponding to the row.
    This function takes in the lat/long and year values for a given row in the DHS CSV and then returns
    the corresponding query we want to send to Google BigQuery corresponding to that row.

    NOTE: for now, we just query the actual lat/long values given. Soon, we'll adjust this to query for a range of values
    within some square radius.
    """

    return f"""
    SELECT DOCUMENTIDENTIFIER
    FROM `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE _PARTITIONTIME >= "{year-1}-01-01 00:00:00" AND
          _PARTITIONTIME < "{year}-01-01 00:00:00" AND
          SOURCECOLLECTIONIDENTIFIER = 1 AND
          REGEXP_CONTAINS(V2LOCATIONS, r'.#{lat}#{long}#.')
    """


def query_for_whole_dataset():
    # get a bigquery client
    client = bigquery.Client()

    dhs_labels_csv = open(PATH_TO_DHS_LABELS, 'r')
    dhs_labels_csv_reader = csv.reader(dhs_labels_csv, delimiter=',')
    for dhs_label_row in dhs_labels_csv_reader:
        curr_query = get_query_string(float(dhs_label_row[3]), float(dhs_label_row[4]), float(dhs_label_row[2]))

        query_result = client.query(curr_query)
        for result_row in query_result:
            pass

    dhs_labels_csv.close()




if __name__ == '__main__':
    query_for_whole_dataset()

















































# SELECT V2Locations FROM `gdelt-bq.gdeltv2.gkg` LIMIT 1000
"""
Notes on the GDELT query that we need:


Fields we'll definitely use
- V2DOCUMENTIDENTIFIER: this is what we'll actually want the query to SELECT. It will give us a means of accessing
the document that the information for the entry (row) in the database corresponds to

- V2SOURCECOLLECTIONIDENTIFIER: tells us what kind of data to expect in v2documentidentifier.
 I think we should only accept a value of 1 for this field (indicating the document identifier is a URL that will
 give us the article with an HTTP GET request.) So, we should specify V2SOURCECOLLECTIONIDENTIFIER=1

- V1LOCATIONS: ***THE MOST IMPORTANT FIELD***
  This field gives the locations listed in the article. LIke other fields, it's semi-colon separated blocks and
  pound separated fields. For example:
  1#California, USA#12#20#-30.1123#40.2341#56;2#Lebanon#01#97#112.2112#51.2#71
  or
  2#Kentucky, United States#US#USKY##37.669#-84.6514#KY#307;2#Kentucky, United States#US#USKY##37.669#-84.6514#KY#1426;2#Kentucky, United States#US#USKY##37.669#-84.6514#KY#1559
  Uh oh; this will be extremely difficult to query.


Fields we might use
- V1THEMES: this basically gives the themes in the document delimited by semi-colons
  You can see a list of the themes here: https://docs.google.com/spreadsheets/d/1oIepKfy8-LNxfEGefy4ViSqLLILQgwod9x9eOy95GJM/edit#gid=772868915
  NOTE that some of these involve poverty!!! This may provide very helpful signal!
  Of course, we shouldn't limit the articles we take too too much.

- V1COUNTS: basically gives counts of various things that happen in the document. It's a list of them. For instance,
  a given row might have V1COUNTS value:
  KILL#47#people#<location_info>;ARREST#3#suspects#<location_info>
  We could use this to acquire extra signal about the article, perhaps; but, I think this might be overkill.




We should query the partitioned gkg table.
Paritioned tables on Google BigQuery basically have pseudo-columns which you can query and when you query on these columns,
you will ONLY execute the query over the rows which satisfy the WHERE condition involving those pseudo-columns.
So, basically, you can improve query performance.
"""

"""
We may want a query like so:
SELECT V2DOCUMENTIDENTIFIER, V1LOCATIONS
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= "2020-01-01 00:00:00" AND   (note: we can use any years for parition time, here; 2020 and 2021 are examples
      _PARTITIONTIME < "2021-01-01 00:00:00" AND
      V2SOURCECOLLECTIONIDENTIFIER = 1 AND          (make sure it's a web source)

I wish we could use some WHERE clause over V1LOCATIONS, but it's basically impossible due to how they structure it.

Here's a real query that worked:
SELECT DOCUMENTIDENTIFIER, V2LOCATIONS
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= "2020-12-30 00:00:00" AND
      _PARTITIONTIME < "2021-01-01 00:00:00" AND
      SOURCECOLLECTIONIDENTIFIER = 1


We probably want to have a condition over V2LOCATIONS where we say it's LIKE (some regex that indicates that it
has the lat/long value desired).
"""



"""
some values for js fiddle test function:

CREATE TABLE test (
   V2LOCATIONS char(200)
)

INSERT INTO test VALUES ("2#cali#US#USCA##40#35#CA#307;5#sumn else#y#n##41#33#y#50)
#INSERT INTO test VALUES ("2#cali#US#USCA#n#40#35#CA#307;5#sumn else#y#n#n#41#33#y#50")
# INSERT INTO test VALUES ("2#Kentucky, United States#US#USKY##37.669#-84.6514#KY#307;2#Kentucky, United States#US#USKY##37.669#-84.6514#KY#1426;2#Kentucky, United States#US#USKY##37.669#-84.6514#KY#1559")

CREATE TABLE test (
   V2LOCATIONS char(200)
);
INSERT INTO test (V2LOCATIONS) VALUES ("hi")








CREATE TABLE test (
   V2LOCATIONS char(200)
);
INSERT INTO test VALUES ("2#cali#US#USCA##40#35#CA#307;5#sumn else#y#n##41#33#y#50");

SELECT STRING_SPLIT(V2LOCATIONS,";")
FROM test
WHERE V2LOCATIONS LIKE "%40#35%"





SELECT DOCUMENTIDENTIFIER, SPLIT(split_locations, "#") AS double_split_locations
FROM (SELECT DOCUMENTIDENTIFIER, SPLIT(V2LOCATIONS, ";") AS split_locations
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= "2020-12-30 00:00:00" AND
      _PARTITIONTIME < "2021-01-01 00:00:00" AND
      SOURCECOLLECTIONIDENTIFIER = 1) AS My_Table


Here's a functioning query:
SELECT DOCUMENTIDENTIFIER, V2LOCATIONS
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= "2020-12-30 00:00:00" AND
      _PARTITIONTIME < "2021-01-01 00:00:00" AND
      SOURCECOLLECTIONIDENTIFIER = 1 AND
      REGEXP_CONTAINS(V2LOCATIONS, r'.#30.2266#-93.2174#.')
BigQuery lets us use regular expressions, so that could be our starting point.
The only thing is, now we still need to run one query per lat/long value, which is obviously not ideal.
Another idea would be to just take all the queries for whatever the year is and just use those.
Let's just write some code to do the querying with arbitrary partitioned dates and with a single lat/long value.
"""



