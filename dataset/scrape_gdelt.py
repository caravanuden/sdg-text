"""
This file queries for GDELT articles relating to the articles we're given

IMPORTANT NOTE: Before using this script, you must follow the instructions given at the top of
configure_google_service_account.sh and you should probably run that script as well
"""
from Utility import *
from google.cloud import bigquery
import csv
from urllib.request import urlopen
from bs4 import BeautifulSoup
import os
import argparse


LAT_AND_LONG_SQUARE_LENGTH = 4

def clean_article(text):
    """

    :param text (string): the article text
    :return: (string) the cleaned article text
    """
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    # anything else we should do?

    return text

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

    def get_all_values_in_square(num):
        return "\\.?.*|".join([str(i) for i in range(int(num) - LAT_AND_LONG_SQUARE_LENGTH, int(num) + LAT_AND_LONG_SQUARE_LENGTH + 1)]) + "\\.?.*"

    # note: we only query on the partition time from year-01-01 to year-02-01 (so only over one month instead of
    # a full year) because then we go over our quota for queries.
    return f"""
    SELECT DOCUMENTIDENTIFIER
    FROM `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE _PARTITIONTIME >= "{year}-01-01 00:00:00" AND
          _PARTITIONTIME < "{year}-02-01 00:00:00" AND
          SOURCECOLLECTIONIDENTIFIER = 1 AND
          REGEXP_CONTAINS(V2LOCATIONS, r\'.#({get_all_values_in_square(lat)})#({get_all_values_in_square(long)})#.\')
    """

def extract_article_text_from_url(url):
    """

    :param url (string): the url for the article we obtained from the query over the GDELT GKG
    :return: (string) the text of the article that's returned.

    NOTE: got this functionality from stack overflow!
    See here: https://stackoverflow.com/questions/328356/extracting-text-from-html-file-using-python
    """
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # rip it out

    return clean_article(soup.get_text())


def query_gdelt_values(dhs_label_rows, out_file_path):
    # get a bigquery client
    client = bigquery.Client()

    articles = dict()

    for i,dhs_label_row in enumerate(dhs_label_rows):
        if i==0:
            continue # this is because hte first row is just the column names.
        curr_query = get_query_string(float(dhs_label_row[3]), float(dhs_label_row[4]), int(dhs_label_row[2]))

        query_job = client.query(curr_query)
        result_rows = query_job.result()
        for result_row in result_rows:
            try:
                article_text = extract_article_text_from_url(result_row[0])

                # now, we need to figure out how to save the article text so we know what row it corresponds to
                # let's use the ID, yeah? given by dhs_label_row[0]
                articles[dhs_label_row[0]] = article_text
            except Exception as e:
                print(f"Got exception getting article text for url {result_row[0]} and exception {e}")


    dhs_labels_csv.close()

    # write the articles
    writeToJsonFile(articles, out_file_path)

    #zip it
    os.system("zip {} {}", out_file_path + ".zip",out_file_path)
    os.system("rm {}", out_file_path)




if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_number', nargs='?', type=int, default=0)
    parser.add_argument('--total_num_processes', nargs='?', type=int, default=1)
    args = parser.parse_args()


    # get the dhs labels
    dhs_labels_csv = open(PATH_TO_DHS_LABELS, 'r')
    dhs_labels_csv_reader = csv.reader(dhs_labels_csv, delimiter=',')
    dhs_labels_rows = [row for row in dhs_labels_csv_reader]

    # split the workload among processes
    rows_to_process = dhs_labels_rows[len(dhs_labels_rows)//(args.total_num_processes) * args.process_number : len(dhs_labels_rows)//(args.total_num_processes) * (args.process_number+1)]

    # complete the workload for this process
    out_file_path = os.path.join(PATH_TO_GDELT_OUTPUTS, f"articles_{args.process_number}")
    query_gdelt_values(rows_to_process, out_file_path)


















































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






SELECT DOCUMENTIDENTIFIER, V2LOCATIONS
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= "2020-12-30 00:00:00" AND
      _PARTITIONTIME < "2021-01-01 00:00:00" AND
      SOURCECOLLECTIONIDENTIFIER = 1 AND
      REGEXP_CONTAINS(V2LOCATIONS, r'.#(30\.?.*|31\.?.*)#-93.2174#.')
      -- REGEXP_CONTAINS(V2LOCATIONS, r'.#30\.+#-93.2174#.') --
    -- REGEXP_CONTAINS(V2LOCATIONS, r'.#(30\.?|31\.?)#-93.2174#.') --
      -- REGEXP_CONTAINS(V2LOCATIONS, r'.#(30\.2266|22)#-93.2174#.') --


      SELECT DOCUMENTIDENTIFIER, DATE
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= "2016-01-01 00:00:00" AND
_PARTITIONTIME < "2016-02-01 00:00:00" AND
SOURCECOLLECTIONIDENTIFIER = 1 AND
REGEXP_CONTAINS(V2LOCATIONS, r'.#(36\.?.*)#.')
"""



